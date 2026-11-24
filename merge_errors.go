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

import "fmt"

// Errors returned by the index consolidation (alias merge) machinery.
// Callers can use == on these sentinels to distinguish failure kinds.
var (
	// ErrorMergeAlreadyRunning is returned when a second consolidation is
	// started on an alias that already has one in progress.  A given alias
	// permits at most one consolidation at any point in time.
	ErrorMergeAlreadyRunning = fmt.Errorf("merge: a consolidation is already running for this alias")

	// ErrorMergeNotRunning is returned when an operation (for example Cancel)
	// targets an alias with no running consolidation.
	ErrorMergeNotRunning = fmt.Errorf("merge: no consolidation is running for this alias")

	// ErrorViewClosed is returned when a consolidated view is used after it
	// has been released.
	ErrorViewClosed = fmt.Errorf("merge: view has been released")

	// ErrorMergeStateCorrupt is returned when a persisted consolidation state
	// file cannot be parsed or fails its integrity checks.
	ErrorMergeStateCorrupt = fmt.Errorf("merge: persisted consolidation state is corrupt")

	// ErrorMergeTargetIncompatible is returned when the target index cannot be
	// used with the source indexes (missing, unopenable, or built with a
	// different mapping).
	ErrorMergeTargetIncompatible = fmt.Errorf("merge: target index is incompatible with the source indexes")
)

// MergePhase identifies the phase a consolidation failed (or is currently
// running) in.  It is carried by MergeError so upper layers can react to the
// failing stage without parsing error strings.
type MergePhase string

const (
	// MergePhasePrepare covers snapshot acquisition, target creation/opening
	// and state file initialization.
	MergePhasePrepare MergePhase = "prepare"

	// MergePhaseApply covers the batched, resumable copy of view documents
	// into the target index.
	MergePhaseApply MergePhase = "apply"

	// MergePhaseCatchup covers the final read of the latest source members to
	// absorb writes that arrived while the apply phase was running.
	MergePhaseCatchup MergePhase = "catchup"

	// MergePhaseVerify covers the pre-switch consistency checks.
	MergePhaseVerify MergePhase = "verify"

	// MergePhaseSwitch covers the atomic alias swap.
	MergePhaseSwitch MergePhase = "switch"
)

// MergeError wraps a consolidation failure with the phase it occurred in.
// Use errors.As to extract it, and errors.Is against the sentinel errors
// above to classify the cause.
type MergeError struct {
	// Phase is the stage the consolidation was in when the error happened.
	Phase MergePhase

	// Op is an optional short description of the exact operation that failed.
	Op string

	// Err is the underlying error.
	Err error
}

func (e *MergeError) Error() string {
	if e.Op != "" {
		return fmt.Sprintf("merge: phase %s, op %s: %v", e.Phase, e.Op, e.Err)
	}
	return fmt.Sprintf("merge: phase %s: %v", e.Phase, e.Err)
}

func (e *MergeError) Unwrap() error {
	return e.Err
}

// newMergeError attaches phase information to an error.  Nil errors pass
// through unchanged so callers can use it in a return expression.
func newMergeError(phase MergePhase, op string, err error) error {
	if err == nil {
		return nil
	}
	return &MergeError{Phase: phase, Op: op, Err: err}
}
