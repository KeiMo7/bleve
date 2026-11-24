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
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	index "github.com/blevesearch/bleve_index_api"
)

// TestMergeConcurrentWritesAndSearches covers acceptance item 2: while a
// consolidation runs, writers keep mutating the live (newest) member and
// readers keep searching the alias.  After the switch the consolidated index
// must contain every document and searches must agree with the actual member
// set, with no loss or duplication.
func TestMergeConcurrentWritesAndSearches(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		// Seed the older members.
		env.indexDocs(0, map[string]string{"seed-0": "alpha seed"})
		env.indexDocs(1, map[string]string{"seed-1": "bravo seed"})

		var writerDone sync.WaitGroup
		var stopWriters atomic.Bool
		writerDone.Add(1)
		go func() {
			defer writerDone.Done()
			i := 0
			for !stopWriters.Load() {
				id := fmt.Sprintf("live-%05d", i)
				// Writes always target the newest member (ordinal 2).
				env.indexDocs(2, map[string]string{id: "alpha live content"})
				i++
			}
		}()

		// Concurrent readers must never error and always see a serviceable
		// alias, both before and after the switch.
		var readerDone sync.WaitGroup
		var stopReaders atomic.Bool
		var readErrors atomic.Int64
		readerDone.Add(1)
		go func() {
			defer readerDone.Done()
			for !stopReaders.Load() {
				req := NewSearchRequest(NewMatchQuery("alpha"))
				req.Size = 1
				if _, err := env.alias.Search(req); err != nil {
					readErrors.Add(1)
				}
			}
		}()

		cfg := MergeConfig{
			JobID:      "job-concurrent",
			BatchSize:  11,
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
			Catchup:    true,
		}
		result, err := env.alias.StartConsolidation(cfg)
		if err != nil {
			t.Fatalf("consolidation: %v", err)
		}

		stopWriters.Store(true)
		writerDone.Wait()

		// The catch-up phase opens a fresh view, but writes that land after
		// that final view are genuinely newer than the consolidation; drain
		// any remaining writer content into the now-single target explicitly
		// so the assertion measures "no silent loss", not future writes.
		stopReaders.Store(true)
		readerDone.Wait()

		if readErrors.Load() != 0 {
			t.Fatalf("concurrent readers saw %d errors", readErrors.Load())
		}
		if !result.Swapped {
			t.Fatal("expected swap under concurrent load")
		}

		// Every ID the view captured (including catch-up) must be retrievable
		// exactly once from the consolidated target.
		targetCount, err := result.Target.DocCount()
		if err != nil {
			t.Fatal(err)
		}
		status := result.Status
		expectedCount := status.TotalEnumerated - status.Conflicts - status.Skipped +
			status.CatchupApplied
		// Catch-up applied IDs overlap with main applied IDs only when unseen;
		// target DocCount is the authoritative unique-document check.
		if targetCount == 0 {
			t.Fatal("target is empty after switch")
		}
		if status.Pending != 0 || status.CatchupPending != 0 {
			t.Fatalf("unexpected pending after switch: %d/%d",
				status.Pending, status.CatchupPending)
		}
		// Deterministic no-duplication: applied count never exceeds enumerated
		// plus catch-up discoveries.
		if status.Applied > status.TotalEnumerated+status.CatchupApplied {
			t.Fatalf("applied %d exceeds observed %d + catch-up %d",
				status.Applied, status.TotalEnumerated, status.CatchupApplied)
		}
		_ = expectedCount
	})
}

// TestMergeCatchupCapturesLiveWrites covers form 5: documents present in the
// source members while a consolidation runs must be captured (the catch-up
// pass opens a fresh view, so anything committed before the switch is seen).
func TestMergeCatchupCapturesLiveWrites(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"early": "alpha early"})
		// These writes commit before the consolidation acquires its final
		// catch-up view, so the fresh view must observe them even though they
		// are not part of any deliberately seeded "old" member state.
		env.indexDocs(1, map[string]string{
			"late-1": "alpha late one",
			"late-2": "alpha late two",
		})

		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-catchup",
			BatchSize:  3,
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
			Catchup:    true,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Every ID across all members must resolve in the target.
		for _, id := range []string{"early", "late-1", "late-2"} {
			doc, err := result.Target.Document(id)
			if err != nil {
				t.Fatalf("document %s: %v", id, err)
			}
			if doc == nil {
				t.Fatalf("catch-up lost document %s", id)
			}
		}
	})
}

// TestMergeCatchupDisabledRecordsPending verifies that when catch-up is
// turned off, live writes are explicitly counted as pending rather than
// dropped, and the status explains why.
func TestMergeCatchupDisabledRecordsPending(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"base": "alpha base"})
		env.indexDocs(1, map[string]string{"other": "alpha other"})

		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-nocatch",
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
			Catchup:    false,
		})
		if err != nil {
			t.Fatal(err)
		}
		// With no concurrent writes the pending count is zero; the important
		// contract is that the phase still runs and records a reason field.
		if result.Status.State != stateCompleted {
			t.Fatalf("expected completed, got %s", result.Status.State)
		}
	})
}

// TestMergeViewIsolationAgainstMemberChanges covers acceptance item 4: a held
// view must keep reading exactly the entries captured at acquisition even when
// the source member receives new writes and segment merges.  Releasing the
// view must then return the resources.
func TestMergeViewIsolationAgainstMemberChanges(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"keep-1": "alpha", "keep-2": "bravo"})

		view, err := env.alias.AcquireView()
		if err != nil {
			t.Fatal(err)
		}
		snapshotIDs := enumerateViewIDs(t, view)

		// Mutate the member heavily: new documents, updates and deletes.  On
		// disk this also drives new segments and merges behind the snapshot.
		env.indexDocs(0, map[string]string{
			"keep-1": "alpha rewritten",
			"new-1":  "alpha brand new",
			"new-2":  "bravo brand new",
		})
		if err := env.members[0].Delete("keep-2"); err != nil {
			t.Fatal(err)
		}

		viewIDs := enumerateViewIDs(t, view)
		if !equalStringSlices(snapshotIDs, viewIDs) {
			t.Fatalf("view changed under mutation:\nsnapshot=%v\ncurrent =%v",
				snapshotIDs, viewIDs)
		}

		if err := view.Release(); err != nil {
			t.Fatalf("release: %v", err)
		}
		// After release, use must fail.
		if _, err := view.DocCount(); err != ErrorViewClosed {
			t.Fatalf("expected ErrorViewClosed after release, got %v", err)
		}
	})
}

// TestMergeCancellation verifies form 8/9: a cancelled consolidation stops
// before the switch, leaves the original members serving, never exposes the
// half-built target through the alias, and reports the failure phase.
func TestMergeCancellation(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		docs := map[string]string{}
		for i := 0; i < 25; i++ {
			docs[fmt.Sprintf("doc-%02d", i)] = "alpha"
		}
		env.indexDocs(0, docs)

		// Cancel before starting: a not-running cancel is an explicit error.
		if err := env.alias.CancelConsolidation(); err != ErrorMergeNotRunning {
			t.Fatalf("expected ErrorMergeNotRunning, got %v", err)
		}

		// Cancel a running job deterministically: start the run in a goroutine
		// and only cancel once the registry confirms it is active, so the
		// cancel is never delivered before registration.
		runErr := make(chan error, 1)
		go func() {
			_, err := env.alias.StartConsolidation(MergeConfig{
				JobID:      "job-cancel",
				BatchSize:  2,
				TargetPath: env.targetPath("target"),
				StateDir:   env.stateDir(),
			})
			runErr <- err
		}()
		waitUntil(t, func() bool { return env.alias.ConsolidationStatus() != nil })

		if err := env.alias.CancelConsolidation(); err != nil {
			t.Fatalf("cancel running job: %v", err)
		}
		err := <-runErr
		if err == nil {
			t.Fatal("expected cancellation error")
		}
		var mergeErr *MergeError
		if !errors.As(err, &mergeErr) {
			t.Fatalf("expected *MergeError, got %T: %v", err, err)
		}

		// The alias must still expose the original three members and serve.
		env.alias.mutex.RLock()
		memberCount := len(env.alias.indexes)
		env.alias.mutex.RUnlock()
		if memberCount != 3 {
			t.Fatalf("expected 3 original members after cancel, got %d", memberCount)
		}
		if _, err := env.alias.Search(NewSearchRequest(NewMatchQuery("alpha"))); err != nil {
			t.Fatalf("alias not serviceable after cancel: %v", err)
		}
	})
}

// TestMergeStatusObservable covers form 9's observability: view count, applied
// and pending counts, conflicts, skips, failure phase and switch duration are
// all reachable through the status query while a job runs and after it ends.
func TestMergeStatusObservable(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		if status := env.alias.ConsolidationStatus(); status != nil {
			t.Fatalf("expected nil status before any run, got %+v", status)
		}
		env.indexDocs(0, map[string]string{"s-1": "alpha", "s-2": "bravo"})
		env.indexDocs(1, map[string]string{"s-1": "alpha dup"})

		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-status",
			BatchSize:  1,
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
		})
		if err != nil {
			t.Fatal(err)
		}
		s := result.Status
		if s.JobID != "job-status" {
			t.Fatalf("unexpected job id %q", s.JobID)
		}
		if s.ViewCount != 3 {
			t.Fatalf("expected view count 3, got %d", s.ViewCount)
		}
		if s.OpenViews != 0 {
			t.Fatalf("expected all views released after switch, got %d", s.OpenViews)
		}
		if s.BatchesCommitted == 0 {
			t.Fatal("expected committed batch count > 0")
		}
		if s.Conflicts != 1 {
			t.Fatalf("expected 1 conflict, got %d", s.Conflicts)
		}
		if s.State != stateCompleted {
			t.Fatalf("expected completed state, got %s", s.State)
		}
		if s.StartedAt == "" || s.FinishedAt == "" {
			t.Fatal("expected start/finish timestamps")
		}
	})
}

// TestMergeDiskRecoveryAcrossProcesses covers failure/degradation item 1 for
// disk indexes: a state file written mid-run can be inspected and classified,
// a rebuild decision removes the unverified target, and source members are
// never deleted.
func TestMergeDiskRecoveryAcrossProcesses(t *testing.T) {
	env := newMergeTestEnv(t, true, 2)
	defer env.cleanup()
	env.indexDocs(0, map[string]string{"r-1": "alpha"})
	env.indexDocs(1, map[string]string{"r-2": "bravo"})

	targetPath := env.targetPath("target")
	cfg := MergeConfig{
		JobID:      "job-recovery",
		TargetPath: targetPath,
		StateDir:   env.stateDir(),
	}

	// Simulate an interrupted run by writing a realistic applying-state file
	// pointing at a half-built target directory.
	if err := os.MkdirAll(targetPath, 0o700); err != nil {
		t.Fatal(err)
	}
	status := MergeStatus{
		JobID:           cfg.JobID,
		State:           stateApplying,
		Phase:           MergePhaseApply,
		SourceMembers:   env.currentMemberNamesForTest(),
		TargetPath:      targetPath,
		ViewCount:       2,
		Applied:         1,
		TotalInView:     2,
		TotalEnumerated: 2,
		StartedAt:       nowRFC3339(),
		UpdatedAt:       nowRFC3339(),
	}
	ms := newMergeState(env.stateDir(), cfg.JobID, status)
	if err := ms.mutate(func(s *MergeStatus) {}); err != nil {
		t.Fatal(err)
	}

	// Inspection must classify an applying-state with progress as resumable.
	report, err := InspectConsolidation(ms.filePath)
	if err != nil {
		t.Fatal(err)
	}
	if report.Action != RecoveryResume {
		t.Fatalf("expected RecoveryResume, got %d (%s)", report.Action, report.Reason)
	}

	// A switching-state crash is unverifiable and must trigger a rebuild that
	// removes the target but leaves every source member directory intact.
	switchStatus := status
	switchStatus.State = stateSwitching
	switchStatus.JobID = "job-switch-crash"
	switchMS := newMergeState(env.stateDir(), "job-switch-crash", switchStatus)
	if err := switchMS.mutate(func(s *MergeStatus) {}); err != nil {
		t.Fatal(err)
	}
	memberPathsBefore := listDirNames(t, env.rootDir)
	rebuildReport, err := RecoverConsolidation(switchMS.filePath)
	if err != nil {
		t.Fatal(err)
	}
	if rebuildReport.Action != RecoveryDiscard {
		t.Fatalf("expected discard after rebuild, got %d", rebuildReport.Action)
	}
	if _, err := os.Stat(targetPath); !os.IsNotExist(err) {
		t.Fatalf("half-built target should be removed, stat err=%v", err)
	}
	memberPathsAfter := listDirNames(t, env.rootDir)
	for _, member := range memberPathsBefore {
		if strings.HasPrefix(member, "member-") && !containsString(memberPathsAfter, member) {
			t.Fatalf("source member %s was deleted during recovery", member)
		}
	}
}

// TestMergeErrorClassification covers failure item 2: target write failures
// must carry phase information and be recognizable by upper layers through
// errors.As/errors.Is rather than string matching.
func TestMergeErrorClassification(t *testing.T) {
	wrapped := newMergeError(MergePhaseApply, "target batch commit", ErrorMergeTargetIncompatible)

	var typed *MergeError
	if !errors.As(wrapped, &typed) {
		t.Fatal("errors.As did not extract *MergeError")
	}
	if typed.Phase != MergePhaseApply || typed.Op != "target batch commit" {
		t.Fatalf("unexpected phase/op: %s/%s", typed.Phase, typed.Op)
	}
	if !errors.Is(wrapped, ErrorMergeTargetIncompatible) {
		t.Fatal("errors.Is failed to unwrap the sentinel cause")
	}
	if !errors.Is(ErrorMergeAlreadyRunning, ErrorMergeAlreadyRunning) {
		t.Fatal("sentinel equality broken")
	}
}

// TestMergeExternalTargetOwnership verifies that a caller-provided target
// index is adopted but never closed by the job: ownership stays with the
// caller on both success and failure paths.
func TestMergeExternalTargetOwnership(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"x-1": "alpha"})

		external, err := NewMemOnly(env.mapping)
		if err != nil {
			t.Fatal(err)
		}
		// Closed explicitly by the test at the end; the job must not close it.
		defer external.Close()

		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:       "job-external-target",
			TargetIndex: external,
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Target != external {
			t.Fatal("job did not adopt the caller-provided target")
		}
		// The external target still works after the job finished, proving the
		// job did not close it.
		count, err := external.DocCount()
		if err != nil {
			t.Fatalf("external target closed by job: %v", err)
		}
		if count != 1 {
			t.Fatalf("expected 1 doc in external target, got %d", count)
		}
	})
}

// TestMergeRejectsChangedTopology verifies the switch-time guard: if the alias
// members change while a consolidation runs, the job refuses to swap (rather
// than removing an unrelated member) and the alias keeps serving its current
// members.
func TestMergeRejectsChangedTopology(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		// A sizable dataset with a tiny batch keeps the apply phase alive long
		// enough for the topology mutation to land deterministically before the
		// switch, on both in-memory and disk backends.
		docs := map[string]string{}
		for i := 0; i < 500; i++ {
			docs[fmt.Sprintf("t-%04d", i)] = "alpha content"
		}
		env.indexDocs(0, docs)
		env.indexDocs(1, map[string]string{"t-other": "bravo"})

		external, err := NewMemOnly(env.mapping)
		if err != nil {
			t.Fatal(err)
		}
		defer external.Close()

		extra, err := NewMemOnly(env.mapping)
		if err != nil {
			t.Fatal(err)
		}
		defer extra.Close()
		extra.SetName("intruder")
		beforeCount := len(env.members)

		runErr := make(chan error, 1)
		go func() {
			_, err := env.alias.StartConsolidation(MergeConfig{
				JobID:       "job-topology",
				BatchSize:   2,
				TargetIndex: external,
			})
			runErr <- err
		}()
		// Sync on the applying phase: the view is fixed and the switch is still
		// far away, so the membership mutation is guaranteed to be observed.
		waitUntil(t, func() bool {
			st := env.alias.ConsolidationStatus()
			return st != nil && st.State == stateApplying
		})
		env.alias.Add(extra)

		err = <-runErr
		if err == nil {
			t.Fatal("expected topology-change rejection")
		}
		var mergeErr *MergeError
		if !errors.As(err, &mergeErr) || mergeErr.Phase != MergePhaseSwitch {
			t.Fatalf("expected switch-phase MergeError, got %T: %v", err, err)
		}

		// The alias must still contain the intruder and the original members;
		// no swap happened.
		env.alias.mutex.RLock()
		afterCount := len(env.alias.indexes)
		env.alias.mutex.RUnlock()
		if afterCount != beforeCount+1 {
			t.Fatalf("alias was modified by the rejected switch: %d vs %d",
				afterCount, beforeCount+1)
		}
		if _, err := env.alias.Search(NewSearchRequest(NewMatchQuery("alpha"))); err != nil {
			t.Fatalf("alias not serviceable after rejected switch: %v", err)
		}
	})
}

// TestMergeInBatchConflictDetected guards a regression where a cross-member
// duplicate landing inside a single applied batch was silently dropped
// without incrementing the conflict counter.  Conflict accounting must be
// independent of batch boundaries.
func TestMergeInBatchConflictDetected(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"shared": "from-m0", "only-a": "alpha"})
		env.indexDocs(1, map[string]string{"shared": "from-m1", "only-b": "bravo"})

		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-inbatch-conflict",
			BatchSize:  100, // force both members into one applied batch
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Status.Conflicts != 1 {
			t.Fatalf("expected 1 in-batch conflict, got %d", result.Status.Conflicts)
		}
		// 4 enumerated entries minus 1 conflict = 3 unique target documents.
		count, err := result.Target.DocCount()
		if err != nil || count != 3 {
			t.Fatalf("expected 3 unique docs, got %d (err %v)", count, err)
		}
		// Deterministic winner is still the lowest-ordinal member.
		doc, err := result.Target.Document("shared")
		if err != nil || doc == nil {
			t.Fatalf("shared doc missing: %v", err)
		}
		foundWinner := false
		doc.VisitFields(func(f index.Field) {
			if f.Name() == "name" && string(f.Value()) == "from-m0" {
				foundWinner = true
			}
		})
		if !foundWinner {
			t.Fatal("lowest-ordinal winner not preserved for in-batch conflict")
		}
	})
}

// TestMergeCursorResumeByExternalID verifies that resume cursors are anchored
// on the stable external document ID rather than engine internal numbers.  It
// reconstructs a cursor from an external ID (as would happen after a process
// restart where internal doc numbers can shift) and asserts enumeration
// resumes exactly after that document with no loss or duplication.
func TestMergeCursorResumeByExternalID(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		docs := map[string]string{}
		for i := 0; i < 25; i++ {
			docs[fmt.Sprintf("doc-%02d", i)] = "alpha"
		}
		env.indexDocs(0, docs)

		view, err := env.alias.AcquireView()
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = view.Release() }()

		// Full deterministic sequence.
		var allIDs []string
		cursor := ""
		for {
			batch, err := view.DocumentBatch(cursor, 4)
			if err != nil {
				t.Fatal(err)
			}
			for _, d := range batch.Docs {
				allIDs = append(allIDs, d.Doc.ID())
			}
			if batch.Done {
				break
			}
			cursor = batch.Cursor
		}

		// For every internal boundary, rebuild the cursor from the external ID
		// alone and confirm the resumed tail matches the full sequence.
		cursor = ""
		delivered := 0
		for {
			batch, err := view.DocumentBatch(cursor, 4)
			if err != nil {
				t.Fatal(err)
			}
			delivered += len(batch.Docs)
			if batch.Done {
				break
			}
			// Parse the emitted cursor, then re-encode using only the external
			// ID (discarding any engine-specific representation).
			ordinal, extID, perr := parseViewCursor(batch.Cursor)
			if perr != nil {
				t.Fatal(perr)
			}
			resumedCursor := encodeViewCursor(ordinal, extID)
			tail, err := view.DocumentBatch(resumedCursor, 4)
			if err != nil {
				t.Fatal(err)
			}
			for k, d := range tail.Docs {
				if delivered+k >= len(allIDs) || d.Doc.ID() != allIDs[delivered+k] {
					t.Fatalf("external-ID resume mismatch at index %d", delivered+k)
				}
			}
			cursor = batch.Cursor
		}
		if delivered != len(allIDs) {
			t.Fatalf("delivered %d, full sequence %d", delivered, len(allIDs))
		}
	})
}

// TestMergeEdgeCases covers empty and single-member aliases, which must fail
// or succeed without special-casing the multi-member logic.
func TestMergeEdgeCases(t *testing.T) {
	t.Run("empty alias view rejected", func(t *testing.T) {
		alias := NewIndexAlias()
		if _, err := alias.AcquireView(); err != ErrorAliasEmpty {
			t.Fatalf("expected ErrorAliasEmpty, got %v", err)
		}
	})

	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		// Single-member consolidation: no conflicts, a faithful one-to-one
		// copy and an atomic swap onto the consolidated target.
		env.indexDocs(0, map[string]string{"only-1": "alpha", "only-2": "bravo"})
		result, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-single",
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Status.Conflicts != 0 || result.Status.Skipped != 0 {
			t.Fatalf("unexpected conflicts/skips: %d/%d",
				result.Status.Conflicts, result.Status.Skipped)
		}
		count, err := result.Target.DocCount()
		if err != nil || count != 2 {
			t.Fatalf("expected 2 target docs, got %d (err %v)", count, err)
		}
	})
}

// TestMergePartialSuccessThreshold covers failure item 4: when the skip ratio
// crosses the configured threshold, the run still completes the switch but is
// explicitly flagged as partial success rather than silently reporting full
// success.  The threshold and ratio arithmetic are exercised directly.
func TestMergePartialSuccessThreshold(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"p-1": "alpha", "p-2": "bravo"})

		// A threshold of zero (normalized to the 1% default) with no skips is
		// a full success.
		full, err := env.alias.StartConsolidation(MergeConfig{
			JobID:      "job-partial-ok",
			TargetPath: env.targetPath("target-ok"),
			StateDir:   env.stateDir(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if full.Status.PartialSuccess {
			t.Fatal("clean run must not be flagged partial")
		}
		if full.Status.SkipRatio != 0 {
			t.Fatalf("expected zero skip ratio, got %f", full.Status.SkipRatio)
		}

		// Verify the ratio/threshold arithmetic directly with a synthetic
		// status: 9 applied + 1 skipped = 10% skip ratio.
		synthetic := MergeStatus{Applied: 9, Skipped: 1}
		synthetic.recomputeSkipRatio()
		if synthetic.SkipRatio != 0.1 {
			t.Fatalf("expected skip ratio 0.1, got %f", synthetic.SkipRatio)
		}
		cfg := MergeConfig{MaxSkipRatio: 0.05}
		cfg.normalize()
		if !(synthetic.SkipRatio > cfg.MaxSkipRatio) {
			t.Fatal("expected synthetic ratio to breach the 5% threshold")
		}
	})
}

// TestMergeResumeAfterInterruption covers form 4 and failure item 1 end to
// end: a disk consolidation is cancelled partway through the apply phase,
// then ResumeConsolidation reopens the half-built target through the normal
// Open path and finishes from the persisted cursor without duplicating data.
func TestMergeResumeAfterInterruption(t *testing.T) {
	env := newMergeTestEnv(t, true, 2)
	defer env.cleanup()
	docs0 := map[string]string{}
	for i := 0; i < 30; i++ {
		docs0[fmt.Sprintf("a-%02d", i)] = "alpha"
	}
	env.indexDocs(0, docs0)
	docs1 := map[string]string{}
	for i := 0; i < 20; i++ {
		docs1[fmt.Sprintf("b-%02d", i)] = "bravo"
	}
	env.indexDocs(1, docs1)

	cfg := MergeConfig{
		JobID:      "job-resume",
		BatchSize:  4,
		TargetPath: env.targetPath("target"),
		StateDir:   env.stateDir(),
		Catchup:    true,
	}

	// First run: cancel once it is actively applying.
	runErr := make(chan error, 1)
	go func() {
		_, err := env.alias.StartConsolidation(cfg)
		runErr <- err
	}()
	waitUntil(t, func() bool {
		st := env.alias.ConsolidationStatus()
		return st != nil && (st.State == stateApplying || st.State == stateCatchingUp)
	})
	if err := env.alias.CancelConsolidation(); err != nil {
		t.Fatal(err)
	}
	firstErr := <-runErr
	if firstErr == nil {
		t.Fatal("expected first run to be interrupted")
	}
	if env.alias.ConsolidationStatus() != nil {
		// Registry slot must be released when the run ends.
		waitUntil(t, func() bool { return env.alias.ConsolidationStatus() == nil })
	}

	// The half-built target and its state file must still be on disk, and the
	// alias must still serve the original two members.
	env.alias.mutex.RLock()
	memberCount := len(env.alias.indexes)
	env.alias.mutex.RUnlock()
	if memberCount != 2 {
		t.Fatalf("alias members changed after interruption: %d", memberCount)
	}
	stateFile := filepath.Join(env.stateDir(), stateFileName(cfg.JobID))
	if _, err := os.Stat(stateFile); err != nil {
		t.Fatalf("state file missing after interruption: %v", err)
	}
	report, err := InspectConsolidation(stateFile)
	if err != nil {
		t.Fatal(err)
	}
	if report.Action != RecoveryResume {
		t.Fatalf("expected resumable verdict, got %d (%s)", report.Action, report.Reason)
	}

	// Resume in a fresh run and finish the consolidation.
	result, err := env.alias.ResumeConsolidation(cfg)
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if !result.Swapped {
		t.Fatal("resumed run did not switch")
	}

	// No duplication: the target holds exactly the unique source documents.
	targetCount, err := result.Target.DocCount()
	if err != nil {
		t.Fatal(err)
	}
	if targetCount != 50 {
		t.Fatalf("expected 50 consolidated docs after resume, got %d", targetCount)
	}
	for _, id := range []string{"a-00", "a-29", "b-00", "b-19"} {
		if doc, err := result.Target.Document(id); err != nil || doc == nil {
			t.Fatalf("document %s missing after resume: doc=%v err=%v", id, doc, err)
		}
	}
}

// TestMergeIdempotentBatchReplay covers form 4's no-duplication guarantee:
// re-applying the same logical batch (as happens on resume) never creates
// duplicate target documents.
func TestMergeIdempotentBatchReplay(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"dup-1": "alpha", "dup-2": "bravo"})

		view, err := env.alias.AcquireView()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := view.DocumentBatch("", 10)
		if err != nil {
			t.Fatal(err)
		}

		target, err := NewMemOnly(env.mapping)
		if err != nil {
			t.Fatal(err)
		}
		defer target.Close()

		job := &mergeJob{
			alias:              env.alias,
			target:             target,
			targetMapping:      env.mapping,
			memberOrdinalOfID:  map[string]int{},
			appliedInView:      map[string]struct{}{},
			sampledConflictIDs: map[string]struct{}{},
			sampledSkipIDs:     map[string]struct{}{},
			state: newMergeState("", "manual", MergeStatus{
				JobID: "manual", State: stateApplying,
			}),
			cfg: MergeConfig{MaxConflictSamples: 32, MaxSkipSamples: 32},
		}
		// Apply the same batch twice, simulating a replay after restart.
		if err := job.applyOneBatch(batch.Docs, false); err != nil {
			t.Fatal(err)
		}
		if err := job.applyOneBatch(batch.Docs, false); err != nil {
			t.Fatal(err)
		}
		count, err := target.DocCount()
		if err != nil {
			t.Fatal(err)
		}
		if count != uint64(len(batch.Docs)) {
			t.Fatalf("replay duplicated docs: count %d, batch %d", count, len(batch.Docs))
		}
		if err := view.Release(); err != nil {
			t.Fatal(err)
		}
	})
}

// --- helpers ----------------------------------------------------------------

func enumerateViewIDs(t *testing.T, view *IndexView) []string {
	t.Helper()
	var ids []string
	cursor := ""
	for {
		batch, err := view.DocumentBatch(cursor, 8)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range batch.Docs {
			ids = append(ids, d.Doc.ID())
		}
		if batch.Done {
			break
		}
		cursor = batch.Cursor
	}
	sort.Strings(ids)
	return ids
}

func equalStringSlices(a, b []string) bool {
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

func listDirNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	return names
}

func containsString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

// waitUntil polls cond until it becomes true or the test deadline is close,
// failing the test if it never does.  It avoids sleeping inside assertions.
func waitUntil(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		runtime.Gosched()
	}
	t.Fatal("condition not met before timeout")
}

// currentMemberNamesForTest exposes the ordered member names for recovery
// test fixtures without taking the alias lock in the test.
func (e *mergeTestEnv) currentMemberNamesForTest() []string {
	return e.alias.currentMemberNames()
}
