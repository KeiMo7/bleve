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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Consolidation lifecycle phases persisted to disk.  The state machine only
// moves forward along this sequence; every transition is fsynced before the
// work of the new phase begins, so a crash always leaves a state that lets a
// later process decide whether to resume, rebuild or discard.
const (
	// statePreparing: target created, view acquired, nothing copied yet.
	statePreparing = "preparing"
	// stateApplying: document batches are being copied into the target.
	stateApplying = "applying"
	// stateCatchingUp: the main copy finished; live writes to source members
	// are being absorbed.
	stateCatchingUp = "catching_up"
	// stateVerifying: pre-switch consistency checks are running.
	stateVerifying = "verifying"
	// stateSwitching: the alias swap is in progress.
	stateSwitching = "switching"
	// stateCompleted: the switch succeeded; the target is now serving.
	stateCompleted = "completed"
	// stateFailed: the consolidation aborted before the switch; the alias is
	// still serving the original members and the target is not queryable.
	stateFailed = "failed"
	// stateDiscarded: the target was torn down after an unrecoverable error.
	stateDiscarded = "discarded"
)

// MergeConflictSample records one deterministic conflict decision.  Only a
// bounded number of samples is retained (see MergeConfig.MaxConflictSamples);
// the full count is always reported through MergeStatus.Conflicts.
type MergeConflictSample struct {
	DocID            string `json:"doc_id"`
	WinnerMember     int    `json:"winner_member"`
	WinnerMemberName string `json:"winner_member_name"`
	LoserMembers     []int  `json:"loser_members"`
}

// MergeSkipSample records one document that could not be copied.
type MergeSkipSample struct {
	DocID  string `json:"doc_id"`
	Member int    `json:"member"`
	Reason string `json:"reason"`
}

// MergeStatus is the single authoritative, externally visible state of a
// consolidation.  Every counter and list here has exactly one writer (the
// merge goroutine) under the job lock, and reads take a consistent snapshot.
type MergeStatus struct {
	// JobID is the unique identifier of this consolidation run.
	JobID string `json:"job_id"`
	// State is the current lifecycle state (see state* constants).
	State string `json:"state"`
	// Phase is the high-level phase the job is in or failed in.
	Phase MergePhase `json:"phase"`

	// SourceMembers is the frozen, ordered member name list the view was
	// acquired over.  It is the authoritative member list for this run.
	SourceMembers []string `json:"source_members"`
	// TargetPath is the target index location ("" for a memory-only target).
	TargetPath string `json:"target_path"`

	// ViewCount is the number of members pinned by the view.
	ViewCount int `json:"view_count"`
	// OpenViews is the number of currently held views (the main view plus any
	// catch-up views), all released on finish.
	OpenViews int `json:"open_views"`

	// TotalInView is DocCount over the view at acquisition time (distinct
	// live documents within each member, summed across members).
	TotalInView uint64 `json:"total_in_view"`
	// TotalEnumerated is the exact number of entries DocIDReaderAll yields
	// across the frozen view.  Cross-member duplicates are counted per
	// occurrence, so TotalEnumerated >= TotalInView.  It is the authoritative
	// denominator for the applied/conflicts/skipped accounting identity.
	TotalEnumerated uint64 `json:"total_enumerated"`
	// Applied is the number of documents successfully written to the target.
	Applied uint64 `json:"applied"`
	// Pending is the number of view documents not yet applied.  It is derived
	// from TotalInView plus catch-up discoveries minus Applied, and always
	// reconciles against the cursor.
	Pending uint64 `json:"pending"`
	// CatchupApplied is documents copied during the catch-up phase.
	CatchupApplied uint64 `json:"catchup_applied"`
	// CatchupPending is documents discovered during catch-up but not yet
	// copied when the job is interrupted.
	CatchupPending uint64 `json:"catchup_pending"`

	// Conflicts counts every ID observed in more than one source member (or
	// clashing with a live write during catch-up).
	Conflicts uint64 `json:"conflicts"`
	// ConflictSamples holds deterministic examples of conflict resolution.
	ConflictSamples []MergeConflictSample `json:"conflict_samples,omitempty"`
	// Skipped counts documents that could not be applied.
	Skipped uint64 `json:"skipped"`
	// SkipSamples holds deterministic examples of skipped documents.
	SkipSamples []MergeSkipSample `json:"skip_samples,omitempty"`

	// BatchesCommitted counts successfully committed target batches.
	BatchesCommitted uint64 `json:"batches_committed"`
	// BatchesRetried counts batch replays after a resumable failure or
	// restart; replay of the same logical batch never duplicates documents
	// because each ID is written at most once per batch and target writes are
	// upserts.
	BatchesRetried uint64 `json:"batches_retried"`

	// Cursor is the resumption token into the view (see IndexView
	// DocumentBatch).  Empty means the main copy is finished.
	Cursor string `json:"cursor"`

	// SkipRatio is Skipped divided by the total observed document count at
	// job end.  Zero until the first document is processed.
	SkipRatio float64 `json:"skip_ratio"`
	// PartialSuccess is true when the job finished but SkipRatio exceeded
	// MergeConfig.MaxSkipRatio.
	PartialSuccess bool `json:"partial_success"`

	// FailurePhase is set when State == stateFailed, naming the phase that
	// failed.
	FailurePhase MergePhase `json:"failure_phase,omitempty"`
	// FailureReason carries the last error message.
	FailureReason string `json:"failure_reason,omitempty"`

	// StartedAt / UpdatedAt / FinishedAt are RFC3339 timestamps.
	StartedAt  string `json:"started_at"`
	UpdatedAt  string `json:"updated_at"`
	FinishedAt string `json:"finished_at,omitempty"`
	// SwitchDuration is the wall-clock duration of the atomic switch, set
	// even when the switch fails.
	SwitchDurationNS int64 `json:"switch_duration_ns"`
}

// mergeState is the on-disk JSON representation plus the in-memory lock.
// Disk layout for a disk-backed job:
//
//	<stateDir>/merge_<jobID>.json        last committed state
//	<stateDir>/merge_<jobID>.json.tmp    staging file, atomically renamed
//
// The rename guarantees that a crash leaves either the previous good state or
// the new one, never a torn file.  Memory-only jobs keep state in memory only
// and are naturally discardable on process exit.
type mergeState struct {
	dir      string
	jobID    string
	filePath string

	mu     sync.Mutex
	status MergeStatus
}

func stateFileName(jobID string) string {
	return "merge_" + jobID + ".json"
}

func newMergeState(dir, jobID string, status MergeStatus) *mergeState {
	ms := &mergeState{
		dir:    dir,
		jobID:  jobID,
		status: status,
	}
	if dir != "" {
		ms.filePath = filepath.Join(dir, stateFileName(jobID))
	}
	return ms
}

// snapshot returns a value copy of the current status, safe for the caller to
// read without holding any lock.
func (ms *mergeState) snapshot() MergeStatus {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	rv := ms.status
	rv.SourceMembers = append([]string(nil), ms.status.SourceMembers...)
	rv.ConflictSamples = append([]MergeConflictSample(nil), ms.status.ConflictSamples...)
	rv.SkipSamples = append([]MergeSkipSample(nil), ms.status.SkipSamples...)
	return rv
}

// mutate applies fn under the state lock, refreshes UpdatedAt and, for disk
// jobs, atomically persists the result before returning.  The in-memory state
// and the on-disk state therefore never diverge across a successful call.
func (ms *mergeState) mutate(fn func(*MergeStatus)) error {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	fn(&ms.status)
	ms.status.UpdatedAt = nowRFC3339()

	if ms.dir == "" {
		return nil
	}
	return ms.persistLocked()
}

// persistLocked writes the status via a temp file + atomic rename.  Callers
// must hold ms.mu.
func (ms *mergeState) persistLocked() error {
	if ms.dir == "" {
		return nil
	}
	if err := os.MkdirAll(ms.dir, 0o700); err != nil {
		return err
	}
	bytes, err := json.MarshalIndent(&ms.status, "", "  ")
	if err != nil {
		return err
	}
	tmpPath := ms.filePath + ".tmp"
	file, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := file.Write(bytes); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, ms.filePath)
}

// loadMergeState reads a previously persisted state file.  It is used both by
// ResumeConsolidation and by the recovery classifier.
func loadMergeState(filePath string) (*MergeStatus, error) {
	bytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	var status MergeStatus
	if err := json.Unmarshal(bytes, &status); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrorMergeStateCorrupt, err)
	}
	if status.JobID == "" {
		return nil, fmt.Errorf("%w: missing job id in %s", ErrorMergeStateCorrupt, filePath)
	}
	return &status, nil
}

// RecoveryAction tells the caller what to do with a consolidation whose
// process died mid-flight.
type RecoveryAction int

const (
	// RecoveryResume: the target contains a prefix of the view and the state
	// cursor is valid; the job can continue from MergeStatus.Cursor.
	RecoveryResume RecoveryAction = iota
	// RecoveryRebuild: partial state exists but cannot be trusted (for
	// example a torn switch); the target must be rebuilt from scratch.
	RecoveryRebuild
	// RecoveryDiscard: the job had already switched or been discarded; no
	// unfinished consolidation remains to recover.
	RecoveryDiscard
)

// classifyRecovery maps a persisted status to a recovery decision and gives a
// human-readable reason.  It never mutates anything; callers drive the actual
// cleanup.
func classifyRecovery(status *MergeStatus) (RecoveryAction, string) {
	switch status.State {
	case stateCompleted:
		return RecoveryDiscard, "consolidation already completed and switched; nothing to resume"
	case stateDiscarded:
		return RecoveryDiscard, "consolidation was already discarded"
	case stateFailed:
		// A failed job never touched the alias; its target is a half-built,
		// non-queryable index.  Resuming is safe because application is
		// cursor-based and idempotent, but only if a cursor exists.
		if status.Cursor != "" || status.Applied > 0 {
			return RecoveryResume, "job failed before switch with resumable progress"
		}
		return RecoveryRebuild, "job failed before any durable progress; rebuild target"
	case stateSwitching:
		// The alias swap is itself in-memory and not observable after process
		// exit, so a crash here means the target's readiness could not be
		// confirmed.  Rebuild to avoid serving an unverified target.
		return RecoveryRebuild, "crash during switch; target readiness unverified"
	case statePreparing, stateApplying, stateCatchingUp, stateVerifying:
		return RecoveryResume, "job interrupted in phase " + status.State + " with durable cursor"
	default:
		return RecoveryRebuild, "unknown state " + status.State
	}
}
