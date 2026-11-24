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
	"fmt"
	"os"
	"path/filepath"
)

// RecoveryReport is the verdict for one interrupted consolidation.
type RecoveryReport struct {
	// JobID identifies the interrupted run.
	JobID string
	// Action is the classified decision.
	Action RecoveryAction
	// Reason explains the decision.
	Reason string
	// StatusPath is the state file the decision was read from.
	StatusPath string
	// TargetPath is the half-built target index, when known.
	TargetPath string
}

// InspectConsolidation reads a persisted state file and classifies it.  It
// performs no mutation, so it is safe to call against a live directory.
func InspectConsolidation(stateFilePath string) (*RecoveryReport, error) {
	status, err := loadMergeState(stateFilePath)
	if err != nil {
		return nil, err
	}
	action, reason := classifyRecovery(status)
	return &RecoveryReport{
		JobID:      status.JobID,
		Action:     action,
		Reason:     reason,
		StatusPath: stateFilePath,
		TargetPath: status.TargetPath,
	}, nil
}

// InspectConsolidationDir scans a directory for merge_*.json state files and
// classifies each.  Unknown/torn files are reported with an error rather than
// aborting the whole scan.
func InspectConsolidationDir(dir string) ([]RecoveryReport, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var reports []RecoveryReport
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || filepath.Ext(name) != ".json" {
			continue
		}
		if !isMergeStateFileName(name) {
			continue
		}
		report, err := InspectConsolidation(filepath.Join(dir, name))
		if err != nil {
			reports = append(reports, RecoveryReport{
				JobID:      name,
				Action:     RecoveryRebuild,
				Reason:     "unreadable state: " + err.Error(),
				StatusPath: filepath.Join(dir, name),
			})
			continue
		}
		reports = append(reports, *report)
	}
	return reports, nil
}

func isMergeStateFileName(name string) bool {
	prefix, suffix := "merge_", ".json"
	return len(name) > len(prefix)+len(suffix) &&
		name[:len(prefix)] == prefix &&
		name[len(name)-len(suffix):] == suffix
}

// RecoverConsolidation resolves an interrupted consolidation on disk:
//
//   - RecoveryResume: nothing is destroyed; the caller should run
//     ResumeConsolidation.
//   - RecoveryRebuild: the unverified/half-built target is removed so a fresh
//     consolidation can create it through the normal open path, and the state
//     file is marked discarded.  The alias (which never saw the target) keeps
//     serving its original members.
//   - RecoveryDiscard: completed/discarded state files are left untouched.
//
// Source member indexes are never deleted by this function.
func RecoverConsolidation(stateFilePath string) (*RecoveryReport, error) {
	report, err := InspectConsolidation(stateFilePath)
	if err != nil {
		return nil, err
	}

	switch report.Action {
	case RecoveryResume, RecoveryDiscard:
		return report, nil
	case RecoveryRebuild:
		if report.TargetPath != "" {
			if err := os.RemoveAll(report.TargetPath); err != nil {
				return report, fmt.Errorf("merge: cannot remove rebuild target %s: %w",
					report.TargetPath, err)
			}
		}
		// Mark the state discarded so repeated recovery is idempotent.
		status, err := loadMergeState(stateFilePath)
		if err != nil {
			return report, err
		}
		status.State = stateDiscarded
		status.FinishedAt = nowRFC3339()
		status.FailureReason = report.Reason
		ms := newMergeState(filepath.Dir(stateFilePath), status.JobID, *status)
		ms.filePath = stateFilePath
		if err := ms.mutate(func(s *MergeStatus) {
			s.State = stateDiscarded
		}); err != nil {
			return report, err
		}
		report.Action = RecoveryDiscard
		report.Reason = "rebuilt target removed; state marked discarded"
		return report, nil
	default:
		return report, fmt.Errorf("merge: unknown recovery action %d", report.Action)
	}
}

// ResumeConsolidation continues a previously interrupted, resumable
// consolidation.  It revalidates the persisted state, reopens the target
// index through the normal Open path and replays from the stored cursor.
// Because application is cursor-based and batch writes are idempotent
// upserts, replaying the last committed batch creates no duplicates.
//
// The cfg must carry the same JobID as the interrupted run; TargetPath is
// read from the state file and must not be changed.  The alias must still
// contain the original source members.
func (i *indexAliasImpl) ResumeConsolidation(cfg MergeConfig) (*MergeResult, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}

	stateDir := cfg.StateDir
	if stateDir == "" && cfg.TargetPath != "" {
		stateDir = filepath.Dir(cfg.TargetPath)
	}
	if stateDir == "" {
		return nil, fmt.Errorf("merge: memory-only consolidations cannot be resumed across processes")
	}

	stateFilePath := filepath.Join(stateDir, stateFileName(cfg.JobID))
	status, err := loadMergeState(stateFilePath)
	if err != nil {
		return nil, err
	}
	action, reason := classifyRecovery(status)
	if action != RecoveryResume {
		return nil, fmt.Errorf("merge: job %s is not resumable: %s", cfg.JobID, reason)
	}

	// Reopen the half-built target through the public Open path.  If it no
	// longer opens cleanly, classify it as needing a rebuild instead of
	// surfacing an opaque engine error to queries.
	target, err := Open(status.TargetPath)
	if err != nil {
		return nil, fmt.Errorf("%w: target cannot be reopened (%v); run RecoverConsolidation to rebuild",
			ErrorMergeTargetIncompatible, err)
	}

	// Re-acquire a fresh view over the current alias members.  The original
	// snapshot cannot survive a process exit, but the deterministic ordering
	// and cursor are defined over member ordinals and document IDs, which
	// remain valid as long as the alias layout is unchanged.
	currentMembers := i.currentMemberNames()
	if !sameOrderedMembers(currentMembers, status.SourceMembers) {
		_ = target.Close()
		return nil, fmt.Errorf("%w: alias layout changed since the interrupted run; expected %v, got %v",
			ErrorMergeTargetIncompatible, status.SourceMembers, currentMembers)
	}

	view, err := i.AcquireView()
	if err != nil {
		_ = target.Close()
		return nil, err
	}

	totalInView, err := view.DocCount()
	if err != nil {
		_ = view.Release()
		_ = target.Close()
		return nil, err
	}

	// Restore the in-memory bookkeeping the conflict/idempotency logic needs.
	// Applied progress is trusted from the persisted state; the cursor guards
	// the resume position.
	job := &mergeJob{
		alias:              i,
		cfg:                cfg,
		view:               view,
		target:             target,
		targetOwned:        false,
		memberOrdinalOfID:  map[string]int{},
		appliedInView:      map[string]struct{}{},
		sampledConflictIDs: map[string]struct{}{},
		sampledSkipIDs:     map[string]struct{}{},
		cancel:             make(chan struct{}),
		done:               make(chan struct{}),
	}
	status.TotalInView = totalInView
	status.OpenViews = 1
	job.targetMapping = target.Mapping()
	resumeState := newMergeState(stateDir, cfg.JobID, *status)
	resumeState.filePath = stateFilePath
	job.setState(resumeState)

	// Rebuild the applied-ID bookkeeping from the half-built target itself.
	// The in-memory maps were lost with the previous process; without this
	// scan the catch-up phase would re-apply already copied documents.
	if err := job.rebuildAppliedSetFromTarget(); err != nil {
		_ = view.Release()
		_ = target.Close()
		return nil, newMergeError(MergePhasePrepare, "rebuild applied set", err)
	}

	globalMergeMu.Lock()
	if existing, ok := runningMergeJobs[i]; ok && existing != nil {
		globalMergeMu.Unlock()
		_ = view.Release()
		_ = target.Close()
		return nil, ErrorMergeAlreadyRunning
	}
	runningMergeJobs[i] = job
	globalMergeMu.Unlock()

	defer func() {
		globalMergeMu.Lock()
		if current := runningMergeJobs[i]; current == job {
			delete(runningMergeJobs, i)
		}
		globalMergeMu.Unlock()
		close(job.done)
	}()

	return job.resumeFromCursor()
}

// resumeFromCursor replays from the persisted cursor and drives the remaining
// phases (catch-up, verify, switch).
func (j *mergeJob) resumeFromCursor() (*MergeResult, error) {
	defer j.cleanupOnExit()

	// Rebuild the applied-ID set up to the cursor by scanning the target
	// would be expensive; instead the cursor guarantees the next view batch
	// starts after every already applied document, and target upserts make a
	// replayed boundary batch harmless.  Mark the run as applying and go.
	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateApplying
		s.Phase = MergePhaseApply
	}); err != nil {
		return nil, err
	}
	if err := j.applyAll(); err != nil {
		j.recordFailure(MergePhaseApply, err)
		return nil, err
	}
	if j.cfg.Catchup {
		if err := j.catchup(); err != nil {
			j.recordFailure(MergePhaseCatchup, err)
			return nil, err
		}
	} else if err := j.markCatchupPending(); err != nil {
		j.recordFailure(MergePhaseCatchup, err)
		return nil, err
	}
	if err := j.verify(); err != nil {
		j.recordFailure(MergePhaseVerify, err)
		return nil, err
	}
	if err := j.switchAlias(); err != nil {
		j.recordFailure(MergePhaseSwitch, err)
		return nil, err
	}
	finalStatus := j.state.snapshot()
	return &MergeResult{Status: finalStatus, Target: j.target, Swapped: true}, nil
}

func (i *indexAliasImpl) currentMemberNames() []string {
	i.mutex.RLock()
	defer i.mutex.RUnlock()
	rv := make([]string, len(i.indexes))
	for idx, member := range i.indexes {
		rv[idx] = member.Name()
	}
	return rv
}

func sameOrderedMembers(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// rebuildAppliedSetFromTarget scans the (half-built) target index and
// repopulates the in-memory applied-ID / member-ordinal maps that conflict
// resolution and catch-up de-duplication rely on.  Every ID already present
// in the target is treated as applied; its winning ordinal is conservatively
// set to 0 because the persisted cursor already guarantees main-copy ordering
// is satisfied — catch-up only needs to know "already applied".
func (j *mergeJob) rebuildAppliedSetFromTarget() error {
	advanced, err := j.target.Advanced()
	if err != nil {
		return err
	}
	reader, err := advanced.Reader()
	if err != nil {
		return err
	}
	defer func() { _ = reader.Close() }()

	docIDReader, err := reader.DocIDReaderAll()
	if err != nil {
		return err
	}
	defer func() { _ = docIDReader.Close() }()

	for {
		internalID, err := docIDReader.Next()
		if err != nil {
			return err
		}
		if internalID == nil {
			break
		}
		externalID, err := reader.ExternalID(internalID)
		if err != nil {
			return err
		}
		j.appliedInView[externalID] = struct{}{}
		if _, known := j.memberOrdinalOfID[externalID]; !known {
			j.memberOrdinalOfID[externalID] = 0
		}
	}
	return nil
}
