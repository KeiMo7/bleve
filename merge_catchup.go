//  Copyright (c) 2024 Couchbase, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 		http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bleve

import (
	"errors"
)

// errMergeCancelled is the internal cause recorded when CancelConsolidation
// stops a run before the switch.
var errMergeCancelled = errors.New("merge: consolidation cancelled before switch")

// catchup absorbs writes that landed in the source members while the main
// copy was running.  A fresh view is opened over the alias' *current*
// members; any ID the main copy never applied is copied deterministically,
// and IDs now duplicated across live members follow the same lowest-ordinal
// winner rule as the main copy.  Nothing observed here can be silently lost:
// anything that cannot be applied is counted in CatchupPending/Skipped.
func (j *mergeJob) catchup() error {
	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateCatchingUp
		s.Phase = MergePhaseCatchup
	}); err != nil {
		return err
	}

	catchupView, err := j.alias.AcquireView()
	if err != nil {
		return newMergeError(MergePhaseCatchup, "acquire catch-up view", err)
	}
	defer func() {
		_ = catchupView.Release()
	}()
	if err := j.state.mutate(func(s *MergeStatus) {
		s.OpenViews = 2
	}); err != nil {
		return err
	}

	// Enumerate the fresh view with the same deterministic batching.  Every
	// unseen ID is a live write; applyOneBatch with catchup=true handles
	// de-duplication against the main-copy set.
	cursor := ""
	var discovered uint64
	for {
		if j.isCancelled() {
			return newMergeError(MergePhaseCatchup, "cancellation", errMergeCancelled)
		}
		batchDocs, err := catchupView.DocumentBatch(cursor, j.cfg.BatchSize)
		if err != nil {
			return newMergeError(MergePhaseCatchup, "read catch-up batch", err)
		}

		newDocs := make([]ViewDocument, 0, len(batchDocs.Docs))
		for _, entry := range batchDocs.Docs {
			if _, applied := j.appliedInView[entry.Doc.ID()]; !applied {
				newDocs = append(newDocs, entry)
				discovered++
			}
		}

		if len(newDocs) > 0 {
			// Temporarily point the batch machinery at the catch-up view for
			// member-name lookups; conflict resolution only needs ordinals,
			// which are identical across views of the same alias layout.
			if err := j.applyCatchupBatch(newDocs); err != nil {
				return err
			}
		}

		cursor = batchDocs.Cursor
		if err := j.state.mutate(func(s *MergeStatus) {
			s.CatchupPending = discovered - s.CatchupApplied
			s.reconcilePending()
		}); err != nil {
			return err
		}
		if batchDocs.Done {
			break
		}
	}

	return j.state.mutate(func(s *MergeStatus) {
		s.OpenViews = 1
		s.CatchupPending = 0
		s.reconcilePending()
	})
}

// applyCatchupBatch runs one catch-up batch, reusing the same deterministic
// conflict rule but accounting into the catch-up counters.  appliedInView is
// the single de-duplication authority, so an ID repeated within the batch is
// handled identically to one repeated across batches.
func (j *mergeJob) applyCatchupBatch(docs []ViewDocument) error {
	targetBatch := j.target.NewBatch()
	if targetBatch == nil {
		return ErrorIndexClosed
	}

	var applied, conflicts, skipped uint64

	for _, entry := range docs {
		docID := entry.Doc.ID()
		if docID == "" {
			skipped++
			j.addSkipSample(MergeSkipSample{Member: entry.SourceMember, Reason: "empty document id"})
			continue
		}
		if _, appliedAlready := j.appliedInView[docID]; appliedAlready {
			// Captured by the main copy, or by an earlier entry in this batch:
			// never replayed, and cross-member collisions were already counted
			// during the main copy's deterministic resolution.
			continue
		}
		if winner, seen := j.memberOrdinalOfID[docID]; seen && winner <= entry.SourceMember {
			// A lower-ordinal live member already wins this ID.
			conflicts++
			j.noteConflict(docID, entry.SourceMember, true)
			continue
		}
		bleveDoc := j.asBleveDocument(entry.Doc)
		if bleveDoc == nil {
			skipped++
			j.addSkipSample(MergeSkipSample{DocID: docID, Member: entry.SourceMember, Reason: "unsupported document type"})
			continue
		}
		if err := targetBatch.IndexAdvanced(bleveDoc); err != nil {
			skipped++
			j.addSkipSample(MergeSkipSample{DocID: docID, Member: entry.SourceMember, Reason: err.Error()})
			continue
		}
		// Record immediately so a duplicate later in the same batch is skipped
		// rather than written twice.
		j.appliedInView[docID] = struct{}{}
		applied++
	}

	if targetBatch.Size() > 0 {
		if err := j.target.Batch(targetBatch); err != nil {
			return newMergeError(MergePhaseCatchup, "target batch commit", err)
		}
	}
	return j.state.mutate(func(s *MergeStatus) {
		s.Applied += applied
		s.CatchupApplied += applied
		s.Conflicts += conflicts
		s.Skipped += skipped
		s.BatchesCommitted++
		s.reconcilePending()
		s.recomputeSkipRatio()
	})
}

// markCatchupPending is used when MergeConfig.Catchup is disabled.  It opens
// a fresh view and explicitly counts documents that appeared after the main
// view was acquired, recording them as CatchupPending instead of dropping
// them silently.  The run then refuses to report unconditional success: the
// caller sees the pending count and the reason in the status.
func (j *mergeJob) markCatchupPending() error {
	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateCatchingUp
		s.Phase = MergePhaseCatchup
	}); err != nil {
		return err
	}

	catchupView, err := j.alias.AcquireView()
	if err != nil {
		return newMergeError(MergePhaseCatchup, "acquire catch-up view", err)
	}
	defer func() {
		_ = catchupView.Release()
	}()

	var pending uint64
	cursor := ""
	for {
		batchDocs, err := catchupView.DocumentBatch(cursor, j.cfg.BatchSize)
		if err != nil {
			return newMergeError(MergePhaseCatchup, "scan catch-up view", err)
		}
		for _, entry := range batchDocs.Docs {
			if _, applied := j.appliedInView[entry.Doc.ID()]; !applied {
				pending++
			}
		}
		cursor = batchDocs.Cursor
		if batchDocs.Done {
			break
		}
	}
	return j.state.mutate(func(s *MergeStatus) {
		s.CatchupPending = pending
		if pending > 0 {
			s.FailureReason = "catch-up disabled; " +
				"documents written during consolidation are recorded as catchup_pending"
		}
		s.reconcilePending()
	})
}

// --- per-alias job registry: status, cancellation, single-flight ------------

// runningJobLocked returns the live job for an alias, or nil.  The caller
// must hold globalMergeMu.
func runningJobLocked(alias *indexAliasImpl) *mergeJob {
	return runningMergeJobs[alias]
}

// ConsolidationStatus returns the authoritative status of the consolidation
// currently running on the alias.  It returns nil when no job is running.
// The returned value is a point-in-time copy and safe to retain.
func (i *indexAliasImpl) ConsolidationStatus() *MergeStatus {
	globalMergeMu.Lock()
	job := runningMergeJobs[i]
	globalMergeMu.Unlock()
	if job == nil {
		return nil
	}
	state := job.getState()
	if state == nil {
		// Job exists but has not finished prepare(); report the bare registry
		// entry so callers still observe a running consolidation.
		return &MergeStatus{State: statePreparing, Phase: MergePhasePrepare, JobID: job.cfg.JobID}
	}
	status := state.snapshot()
	return &status
}

// CancelConsolidation requests cancellation of the running consolidation.
// The request is cooperative: the job stops at the next batch boundary, marks
// itself failed in the pre-switch phase it reached, and leaves the original
// members serving.  It returns ErrorMergeNotRunning when no job is active.
// Cancellation never interrupts the atomic switch once it has started; either
// the full switch completes or it never begins.
func (i *indexAliasImpl) CancelConsolidation() error {
	globalMergeMu.Lock()
	job := runningMergeJobs[i]
	globalMergeMu.Unlock()
	if job == nil {
		return ErrorMergeNotRunning
	}
	job.switchMu.Lock()
	defer job.switchMu.Unlock()
	select {
	case <-job.cancel:
		// already cancelled
	default:
		close(job.cancel)
	}
	return nil
}

func (j *mergeJob) isCancelled() bool {
	select {
	case <-j.cancel:
		return true
	default:
		return false
	}
}

// AwaitConsolidation blocks until the running consolidation finishes (or
// until the optional stop channel closes).  It returns the final status, or
// nil if no job is running.
func (i *indexAliasImpl) AwaitConsolidation(stop <-chan struct{}) *MergeStatus {
	globalMergeMu.Lock()
	job := runningMergeJobs[i]
	globalMergeMu.Unlock()
	if job == nil {
		return nil
	}
	if stop == nil {
		<-job.done
	} else {
		select {
		case <-job.done:
		case <-stop:
		}
	}
	return i.ConsolidationStatus()
}
