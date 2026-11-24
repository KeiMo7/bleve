//  Copyright (c) 2024 Couchbase, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 		http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bleve

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/blevesearch/bleve/v2/analysis"
	"github.com/blevesearch/bleve/v2/document"
	"github.com/blevesearch/bleve/v2/mapping"
	"github.com/blevesearch/bleve/v2/util"
	index "github.com/blevesearch/bleve_index_api"
)

// asBleveDocument narrows the engine-level document returned by a snapshot to
// the concrete *document.Document that Batch.IndexAdvanced expects.  Both the
// scorch and upsidedown (including in-memory gtreap) readers reconstruct
// exactly this type.
//
// Two corrections are applied so the copy is a faithful, searchable clone:
//   - text fields are rebuilt with the target mapping's analyzer, because a
//     snapshot-reconstructed TextField carries no analyzer and would otherwise
//     be indexed as a single whole-value token (breaking normal queries);
//   - the StoreField option is OR-ed in, because the engine's reconstruction
//     constructors default to options without StoreField even for fields that
//     were originally stored.
func (j *mergeJob) asBleveDocument(doc index.Document) *document.Document {
	source, ok := doc.(*document.Document)
	if !ok {
		// Foreign Index implementations are unsupported for consolidation:
		// the public write path needs the concrete bleve document type to
		// preserve stored fields without re-running the mapping over source
		// JSON we do not have.
		return nil
	}
	reb := document.NewDocument(source.ID())
	var hasAllComposite bool
	source.VisitFields(func(field index.Field) {
		rebuilt := j.rebuildField(field)
		if docField, ok := rebuilt.(document.Field); ok {
			reb.AddField(docField)
		}
	})
	source.VisitComposite(func(cf index.CompositeField) {
		if composite, ok := cf.(*document.CompositeField); ok {
			reb.AddField(composite)
			if composite.Name() == allFieldName {
				hasAllComposite = true
			}
		}
	})
	// The engine's analyze() only populates a composite field during
	// re-indexing if the document actually carries one.  Snapshot
	// reconstruction does not return the "_all" composite, so default (no
	// explicit field) queries would otherwise match nothing.  Re-create it
	// with the same options the normal mapping path uses; Compose is then run
	// by the engine when the target batch is analyzed.
	if !hasAllComposite && j.allFieldEnabled() {
		reb.AddField(document.NewCompositeFieldWithIndexingOptions(
			allFieldName, true, []string{}, []string{},
			index.IndexField|index.IncludeTermVectors))
	}
	if nested, ok := interface{}(source).(index.NestedDocument); ok {
		nested.VisitNestedDocuments(func(child index.Document) {
			if childDoc := j.asBleveDocument(child); childDoc != nil {
				reb.AddNestedDocument(childDoc)
			}
		})
	}
	reb.StoredFieldsSize = source.StoredFieldsBytes()
	return reb
}

// allFieldName is the bleve default composite field that aggregates indexed
// text for field-less queries.
const allFieldName = "_all"

// allFieldEnabled reports whether to emit the "_all" composite field.  The
// mapping package's per-field enablement lookup is not exported, so the
// consolidation uses the fresh-IndexMapping default (enabled), which is also
// what every source index in a homogeneous alias uses unless it explicitly
// disabled "_all".
func (j *mergeJob) allFieldEnabled() bool {
	return true
}

// rebuildField re-creates a reconstructed field with an analyzer and the
// stored option so it indexes and stores identically to the source.  Only
// text fields need this treatment; other field types already round-trip
// through their FromBytes constructors.
func (j *mergeJob) rebuildField(field index.Field) index.Field {
	text, ok := field.(*document.TextField)
	if !ok {
		return field
	}
	options := text.Options() | index.StoreField
	analyzer := j.resolveFieldAnalyzer(text.Name())
	return document.NewTextFieldCustom(
		text.Name(), text.ArrayPositions(), text.Value(), options, analyzer)
}

// resolveFieldAnalyzer returns the analyzer the target mapping would apply to
// a field of the given name.  Homogeneous members share one mapping, which is
// enforced in prepare(); the default analyzer is the correct choice for
// dynamically mapped fields and a safe fallback for explicitly mapped ones.
func (j *mergeJob) resolveFieldAnalyzer(_ string) analysis.Analyzer {
	if j.targetMapping == nil {
		return nil
	}
	if impl, ok := j.targetMapping.(*mapping.IndexMappingImpl); ok {
		return impl.AnalyzerNamed(impl.DefaultAnalyzer)
	}
	return j.targetMapping.AnalyzerNamed("")
}

// jsonMarshalMapping serializes a mapping for structural comparison.
func jsonMarshalMapping(m mapping.IndexMapping) ([]byte, error) {
	if m == nil {
		return nil, errors.New("merge: nil index mapping")
	}
	return util.MarshalJSON(m)
}

// MergeConfig configures an alias consolidation run.
type MergeConfig struct {
	// JobID is the caller-chosen unique identifier for this run.  It names the
	// persisted state file, so reusing a JobID reopens (and resumes) the same
	// consolidation rather than starting a new one.  Required.
	JobID string

	// TargetPath is where the consolidated index is created.  An empty string
	// creates a memory-only target, in which case progress cannot survive a
	// process restart and consolidation must complete within one process.
	TargetPath string

	// TargetIndex, when non-nil, is an already created/opened index to write
	// the consolidated data into.  Ownership stays with the caller; the job
	// never closes it.  When nil, the job creates a new index at TargetPath
	// using the source alias' mapping and owns (and closes) it.
	TargetIndex Index

	// BatchSize is the maximum number of documents applied per target Batch.
	// Defaults to 256 when <= 0.
	BatchSize int

	// MaxConflictSamples bounds the retained conflict samples (the conflict
	// count itself is always exact).  Defaults to 32 when < 0.
	MaxConflictSamples int
	// MaxSkipSamples bounds the retained skip samples.  Defaults to 32 when
	// < 0.
	MaxSkipSamples int

	// MaxSkipRatio ends a run with PartialSuccess=true when the share of
	// skipped documents exceeds it.  Defaults to 0.01 (1%).  Set to 1.0 to
	// disable the threshold.
	MaxSkipRatio float64

	// Catchup, when true (default), performs the final live-write pass: after
	// the main copy finishes, fresh views are opened over the source members
	// and every ID not already captured is applied, so writes arriving during
	// the long copy cannot be silently lost.  When false, any such write is
	// explicitly recorded in CatchupPending instead.
	Catchup bool

	// StateDir overrides where the state file is written.  Defaults to
	// TargetPath's parent directory for disk jobs; memory-only jobs keep the
	// state in memory.
	StateDir string
}

// MergeResult summarizes a finished consolidation.
type MergeResult struct {
	// Status is the final authoritative status.
	Status MergeStatus
	// Target is the consolidated index, already swapped into the alias.
	Target Index
	// Swapped is true when the alias was switched in this run.  A
	// PartialSuccess run still swaps (the target holds everything that could
	// be copied); a failed run never swaps.
	Swapped bool
}

// mergeJob is the per-alias orchestrator.  One alias holds at most one live
// job, enforced by the registry in merge_registry.go.
type mergeJob struct {
	alias  *indexAliasImpl
	cfg    MergeConfig
	view   *IndexView
	target Index
	// targetMapping is the mapping used to resolve analyzers when rebuilding
	// reconstructed text fields for the advanced write path.
	targetMapping mapping.IndexMapping

	// stateMu guards the state pointer, which is published by the job
	// goroutine at the end of prepare and read by status queries from other
	// goroutines.  The MergeStatus contents have their own lock inside
	// mergeState.
	stateMu sync.RWMutex
	state   *mergeState
	// targetOwned is true when the job created the target and is responsible
	// for closing/removing it on failure.
	targetOwned bool

	// memberOrdinalOfID records the first (lowest ordinal) member that
	// supplied each ID during the main copy.  Lower ordinal always wins,
	// making the decision independent of map order and scheduling.
	memberOrdinalOfID map[string]int

	// appliedInView records IDs already written to the target during the main
	// copy, used by catch-up to know which live writes are new.  It is keyed
	// by external doc ID.
	appliedInView map[string]struct{}
	// sampledConflictIDs de-duplicates conflict samples by doc ID: the count
	// still records every losing occurrence, but the bounded sample list keeps
	// one entry per conflicted ID.
	sampledConflictIDs map[string]struct{}
	// sampledSkipIDs de-duplicates skip samples analogously.
	sampledSkipIDs map[string]struct{}

	cancel chan struct{}
	done   chan struct{}

	// switchMu serializes Cancel against the swap so a cancellation cannot
	// interleave halfway through the topology change.
	switchMu sync.Mutex
}

// registry: one running job per alias.  Keyed by the alias pointer.
var (
	globalMergeMu    sync.Mutex
	runningMergeJobs = map[*indexAliasImpl]*mergeJob{}
)

// nowRFC3339 is a small indirection so tests can freeze time if needed.
var nowRFC3339 = func() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func (cfg *MergeConfig) normalize() error {
	if cfg.JobID == "" {
		return errors.New("merge: JobID is required")
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 256
	}
	if cfg.MaxConflictSamples <= 0 {
		cfg.MaxConflictSamples = 32
	}
	if cfg.MaxSkipSamples <= 0 {
		cfg.MaxSkipSamples = 32
	}
	if cfg.MaxSkipRatio == 0 {
		cfg.MaxSkipRatio = 0.01
	}
	if cfg.MaxSkipRatio > 1 {
		cfg.MaxSkipRatio = 1
	}
	return nil
}

// StartConsolidation acquires a point-in-time view over the alias and begins
// copying it into a fresh target index.  At most one consolidation may run per
// alias at a time; a duplicate start fails with ErrorMergeAlreadyRunning.
//
// The run executes synchronously and returns when the consolidation either
// switches successfully or fails.  Use ConsolidationStatus to observe progress
// from another goroutine, and CancelConsolidation to stop a run early.
func (i *indexAliasImpl) StartConsolidation(cfg MergeConfig) (*MergeResult, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}

	globalMergeMu.Lock()
	if existing, ok := runningMergeJobs[i]; ok {
		globalMergeMu.Unlock()
		_ = existing
		return nil, ErrorMergeAlreadyRunning
	}
	job := &mergeJob{
		alias:              i,
		cfg:                cfg,
		memberOrdinalOfID:  map[string]int{},
		appliedInView:      map[string]struct{}{},
		sampledConflictIDs: map[string]struct{}{},
		sampledSkipIDs:     map[string]struct{}{},
		cancel:             make(chan struct{}),
		done:               make(chan struct{}),
	}
	runningMergeJobs[i] = job
	globalMergeMu.Unlock()

	defer func() {
		globalMergeMu.Lock()
		// Only remove ourselves; a resumed/replaced job may have taken the
		// slot, but StartConsolidation never replaces a running job.
		if current := runningMergeJobs[i]; current == job {
			delete(runningMergeJobs, i)
		}
		globalMergeMu.Unlock()
		close(job.done)
	}()

	result, err := job.run()
	if err != nil {
		return result, err
	}
	return result, nil
}

// run drives the full lifecycle: prepare -> apply -> catch-up -> verify ->
// switch, with every transition persisted before it is acted on.
func (j *mergeJob) run() (result *MergeResult, err error) {
	defer j.cleanupOnExit()

	if err = j.prepare(); err != nil {
		return nil, err
	}
	if err = j.applyAll(); err != nil {
		j.recordFailure(MergePhaseApply, err)
		return nil, err
	}
	if j.cfg.Catchup {
		if err = j.catchup(); err != nil {
			j.recordFailure(MergePhaseCatchup, err)
			return nil, err
		}
	} else {
		// Explicitly record anything written to a source member after view
		// acquisition as pending rather than silently dropping it.  This is
		// computed during verify against fresh member snapshots.
		if err = j.markCatchupPending(); err != nil {
			j.recordFailure(MergePhaseCatchup, err)
			return nil, err
		}
	}
	if err = j.verify(); err != nil {
		j.recordFailure(MergePhaseVerify, err)
		return nil, err
	}
	if err = j.switchAlias(); err != nil {
		j.recordFailure(MergePhaseSwitch, err)
		return nil, err
	}

	finalStatus := j.state.snapshot()
	return &MergeResult{
		Status:  finalStatus,
		Target:  j.target,
		Swapped: true,
	}, nil
}

// prepare opens the view, creates/opens the target and writes the initial
// state.  Nothing is queryable through the alias at this point.
func (j *mergeJob) prepare() error {
	view, err := j.alias.AcquireView()
	if err != nil {
		return newMergeError(MergePhasePrepare, "acquire view", err)
	}
	j.view = view

	memberNames := make([]string, len(view.members))
	for i, m := range view.members {
		memberNames[i] = m.Name
	}

	totalInView, err := view.DocCount()
	if err != nil {
		return newMergeError(MergePhasePrepare, "view doc count", err)
	}
	totalEnumerated, err := view.EnumeratedCount()
	if err != nil {
		return newMergeError(MergePhasePrepare, "view enumeration count", err)
	}

	stateDir := j.cfg.StateDir
	if stateDir == "" && j.cfg.TargetPath != "" {
		stateDir = filepath.Dir(j.cfg.TargetPath)
	}
	// Memory-only jobs (no path, no state dir) keep state purely in memory.
	if j.cfg.TargetIndex == nil && j.cfg.TargetPath == "" {
		stateDir = ""
	}

	status := MergeStatus{
		JobID:           j.cfg.JobID,
		State:           statePreparing,
		Phase:           MergePhasePrepare,
		SourceMembers:   memberNames,
		TargetPath:      j.cfg.TargetPath,
		ViewCount:       len(view.members),
		OpenViews:       1,
		TotalInView:     totalInView,
		TotalEnumerated: totalEnumerated,
		Pending:         totalInView,
		StartedAt:       nowRFC3339(),
		UpdatedAt:       nowRFC3339(),
	}
	j.setState(newMergeState(stateDir, j.cfg.JobID, status))
	if err := j.persistInitialState(); err != nil {
		return newMergeError(MergePhasePrepare, "persist state", err)
	}

	if err := j.validateMemberMappings(); err != nil {
		return newMergeError(MergePhasePrepare, "validate member mappings", err)
	}

	if err := j.openTarget(); err != nil {
		return newMergeError(MergePhasePrepare, "open target", err)
	}
	return nil
}

// persistInitialState refuses to silently overwrite state belonging to another
// run; resume goes through ResumeConsolidation instead.
func (j *mergeJob) persistInitialState() error {
	if j.state.dir == "" {
		return nil
	}
	if _, err := os.Stat(j.state.filePath); err == nil {
		existing, lerr := loadMergeState(j.state.filePath)
		if lerr != nil {
			return lerr
		}
		if existing.State != stateCompleted && existing.State != stateDiscarded {
			return fmt.Errorf("merge: state file %s already exists for an unfinished job %s; use ResumeConsolidation",
				j.state.filePath, existing.JobID)
		}
	}
	return j.state.mutate(func(s *MergeStatus) {
		s.State = statePreparing
	})
}

// setState publishes the merge state pointer under the state lock.
func (j *mergeJob) setState(ms *mergeState) {
	j.stateMu.Lock()
	j.state = ms
	j.stateMu.Unlock()
}

// getState returns the current merge state pointer, or nil before prepare
// finishes publishing it.
func (j *mergeJob) getState() *mergeState {
	j.stateMu.RLock()
	defer j.stateMu.RUnlock()
	return j.state
}

// sourceMapping returns the mapping to use for target creation.  A
// multi-member alias cannot answer Mapping() directly, so the first view
// member's mapping is used as the canonical mapping.  Consolidation requires
// homogeneous members; prepare() verifies that every member reports the same
// mapping before proceeding.
func (j *mergeJob) sourceMapping() mapping.IndexMapping {
	if j.view != nil && len(j.view.members) > 0 {
		return j.view.members[0].idx.Mapping()
	}
	return j.alias.Mapping()
}

// validateMemberMappings ensures all frozen members share one mapping, which
// is what makes the consolidated target a faithful replacement.  Mappings are
// compared through their serialized form so structural equality does not
// depend on pointer identity.
func (j *mergeJob) validateMemberMappings() error {
	if len(j.view.members) < 2 {
		return nil
	}
	canonical, err := jsonMarshalMapping(j.view.members[0].idx.Mapping())
	if err != nil {
		return err
	}
	for i := 1; i < len(j.view.members); i++ {
		other, err := jsonMarshalMapping(j.view.members[i].idx.Mapping())
		if err != nil {
			return err
		}
		if string(canonical) != string(other) {
			return fmt.Errorf("%w: member %d mapping differs from member 0",
				ErrorMergeTargetIncompatible, i)
		}
	}
	return nil
}

// openTarget creates a brand new index (never reusing an existing queryable
// index directory) or adopts the caller-provided target.
func (j *mergeJob) openTarget() error {
	if j.cfg.TargetIndex != nil {
		j.target = j.cfg.TargetIndex
		j.targetOwned = false
		j.targetMapping = j.cfg.TargetIndex.Mapping()
		return nil
	}

	sourceMapping := j.sourceMapping()
	if sourceMapping == nil {
		return errors.New("merge: alias has no usable mapping for target creation")
	}
	j.targetMapping = sourceMapping

	if j.cfg.TargetPath == "" {
		target, err := NewMemOnly(sourceMapping)
		if err != nil {
			return err
		}
		j.target = target
		j.targetOwned = true
		return nil
	}

	if _, err := os.Stat(j.cfg.TargetPath); err == nil {
		return fmt.Errorf("%w: target path %s already exists", ErrorMergeTargetIncompatible, j.cfg.TargetPath)
	}
	target, err := New(j.cfg.TargetPath, sourceMapping)
	if err != nil {
		return err
	}
	j.target = target
	j.targetOwned = true
	return nil
}

// applyAll streams view documents into the target in deterministic batches.
// Each committed batch updates the cursor before the next batch starts, which
// makes interruption resumable; replaying a batch cannot duplicate data
// because target writes are upserts keyed by document ID and every batch
// deduplicates IDs in memory.
func (j *mergeJob) applyAll() error {
	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateApplying
		s.Phase = MergePhaseApply
	}); err != nil {
		return err
	}

	cursor := j.state.snapshot().Cursor
	for {
		if j.isCancelled() {
			return newMergeError(MergePhaseApply, "cancellation", errMergeCancelled)
		}
		batchDocs, err := j.view.DocumentBatch(cursor, j.cfg.BatchSize)
		if err != nil {
			return newMergeError(MergePhaseApply, "read view batch", err)
		}

		if len(batchDocs.Docs) > 0 || len(batchDocs.SkippedIDs) > 0 {
			if err := j.applyOneBatch(batchDocs.Docs, false); err != nil {
				return err
			}
			if len(batchDocs.SkippedIDs) > 0 {
				if err := j.state.mutate(func(s *MergeStatus) {
					s.Skipped += uint64(len(batchDocs.SkippedIDs))
					for _, skippedID := range batchDocs.SkippedIDs {
						if len(s.SkipSamples) < j.cfg.MaxSkipSamples {
							s.SkipSamples = append(s.SkipSamples, MergeSkipSample{
								DocID: skippedID, Reason: "document unreadable in source snapshot",
							})
						}
					}
					s.recomputeSkipRatio()
				}); err != nil {
					return err
				}
			}
			cursor = batchDocs.Cursor
		}

		// Persist the new cursor (and counters) before the next batch so a
		// crash resumes after the last committed batch.
		if err := j.state.mutate(func(s *MergeStatus) {
			s.Cursor = cursor
			s.reconcilePending()
		}); err != nil {
			return err
		}

		if batchDocs.Done {
			break
		}
	}
	return nil
}

// applyOneBatch turns one view batch into a target Batch.  Deterministic
// conflict rule: when the same external ID appears in multiple source members,
// the member with the lowest ordinal wins; every later member carrying that ID
// contributes exactly one conflict, regardless of batch boundaries.  The
// in-memory memberOrdinalOfID map is the cross-batch memory that makes the
// decision deterministic and independent of map order or scheduling.
//
// catchup=true switches the rule to "target already holds main-copy data; only
// unseen IDs are written, and any ID present in multiple live members is again
// resolved by lowest ordinal".
func (j *mergeJob) applyOneBatch(docs []ViewDocument, catchup bool) error {
	targetBatch := j.target.NewBatch()
	if targetBatch == nil {
		return ErrorIndexClosed
	}

	// memberOrdinalOfID is updated the moment an ID is first staged, so an ID
	// duplicated either within this batch or across batches is detected through
	// one uniform mechanism rather than a separate per-batch set that would
	// miss cross-member conflicts landing in the same batch.
	var batchApplied, batchConflicts, batchSkipped uint64

	for _, entry := range docs {
		docID := entry.Doc.ID()
		if docID == "" {
			batchSkipped++
			j.addSkipSample(MergeSkipSample{Member: entry.SourceMember, Reason: "empty document id"})
			continue
		}

		if !catchup {
			if winner, seen := j.memberOrdinalOfID[docID]; seen {
				// Same ID already supplied by an earlier entry.  Enumeration
				// visits members by ordinal, so the recorded winner is the
				// lower ordinal; this later occurrence is one conflict and is
				// not written again.  The same rule covers duplicates within a
				// single batch because the map is updated on first staging.
				batchConflicts++
				j.noteConflict(docID, entry.SourceMember, true)
				j.recordConflictWinner(docID, winner, entry.SourceMember)
				continue
			}
		} else {
			if _, applied := j.appliedInView[docID]; applied {
				// Main copy already captured this ID; the live write does not
				// need to be replayed.
				continue
			}
			if winner, seen := j.memberOrdinalOfID[docID]; seen && winner <= entry.SourceMember {
				batchConflicts++
				j.noteConflict(docID, entry.SourceMember, true)
				continue
			}
		}

		// Only stored fields can be reconstructed by the engine, so the copy
		// goes through IndexAdvanced: it reuses the analyzed/stored field
		// representation without requiring the original source JSON and
		// without bypassing the target's write path.
		bleveDoc := j.asBleveDocument(entry.Doc)
		if bleveDoc == nil {
			batchSkipped++
			j.addSkipSample(MergeSkipSample{DocID: docID, Member: entry.SourceMember, Reason: "unsupported document type"})
			continue
		}
		if err := targetBatch.IndexAdvanced(bleveDoc); err != nil {
			batchSkipped++
			j.addSkipSample(MergeSkipSample{DocID: docID, Member: entry.SourceMember, Reason: err.Error()})
			continue
		}

		// Record the winning ordinal immediately so any later occurrence in
		// this same batch is correctly treated as a conflict.
		if !catchup {
			j.memberOrdinalOfID[docID] = entry.SourceMember
		}
		j.appliedInView[docID] = struct{}{}
		batchApplied++
	}

	if targetBatch.Size() > 0 {
		if err := j.target.Batch(targetBatch); err != nil {
			// The batch is atomic: nothing partial becomes visible.  Surface
			// the phase so upper layers can classify space/mapping/checksum
			// failures.
			return newMergeError(MergePhaseApply, "target batch commit", err)
		}
	}

	return j.state.mutate(func(s *MergeStatus) {
		s.Applied += batchApplied
		if catchup {
			s.CatchupApplied += batchApplied
		}
		s.Conflicts += batchConflicts
		s.Skipped += batchSkipped
		s.BatchesCommitted++
		s.reconcilePending()
		s.recomputeSkipRatio()
	})
}

// noteConflict records one losing occurrence of docID.  The conflict counter
// counts every losing occurrence (so an ID present in three members yields two
// conflicts), while conflict samples are de-duplicated per ID through
// recordConflictSample.  This keeps the count exact without flooding the
// bounded sample list.
func (j *mergeJob) noteConflict(docID string, memberOrdinal int, retainSample bool) {
	if retainSample {
		j.recordConflictSample(docID, memberOrdinal)
	}
}

// recordConflictWinner stores a sample showing the winning member versus the
// losing member when a cross-member duplicate is detected.
func (j *mergeJob) recordConflictWinner(docID string, winnerOrdinal, loserOrdinal int) {
	j.recordConflictSampleWith(docID, winnerOrdinal, []int{loserOrdinal})
}

func (j *mergeJob) recordConflictSample(docID string, memberOrdinal int) {
	winner := memberOrdinal
	if w, ok := j.memberOrdinalOfID[docID]; ok {
		winner = w
	}
	j.recordConflictSampleWith(docID, winner, []int{memberOrdinal})
}

func (j *mergeJob) recordConflictSampleWith(docID string, winnerOrdinal int, losers []int) {
	if _, seen := j.sampledConflictIDs[docID]; seen {
		return
	}
	status := j.state.snapshot()
	if len(status.ConflictSamples) >= j.cfg.MaxConflictSamples {
		return
	}
	winnerName := ""
	if j.view != nil && winnerOrdinal >= 0 && winnerOrdinal < len(j.view.members) {
		winnerName = j.view.members[winnerOrdinal].Name
	}
	sample := MergeConflictSample{
		DocID:            docID,
		WinnerMember:     winnerOrdinal,
		WinnerMemberName: winnerName,
		LoserMembers:     append([]int(nil), losers...),
	}
	j.sampledConflictIDs[docID] = struct{}{}
	_ = j.state.mutate(func(s *MergeStatus) {
		if len(s.ConflictSamples) < j.cfg.MaxConflictSamples {
			s.ConflictSamples = append(s.ConflictSamples, sample)
		}
	})
}

func (j *mergeJob) addSkipSample(sample MergeSkipSample) {
	if sample.DocID != "" {
		if _, seen := j.sampledSkipIDs[sample.DocID]; seen {
			return
		}
		j.sampledSkipIDs[sample.DocID] = struct{}{}
	}
	status := j.state.snapshot()
	if len(status.SkipSamples) >= j.cfg.MaxSkipSamples {
		return
	}
	_ = j.state.mutate(func(s *MergeStatus) {
		if len(s.SkipSamples) < j.cfg.MaxSkipSamples {
			s.SkipSamples = append(s.SkipSamples, sample)
		}
	})
}

// verify runs the pre-switch checks: target doc count reconciles against
// applied progress, and partial-success threshold is evaluated.  A failed
// check aborts before the alias is ever touched.
func (j *mergeJob) verify() error {
	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateVerifying
		s.Phase = MergePhaseVerify
	}); err != nil {
		return err
	}

	targetCount, err := j.target.DocCount()
	if err != nil {
		return newMergeError(MergePhaseVerify, "target doc count", err)
	}

	status := j.state.snapshot()
	expectedDistinct := status.Applied
	// Applied includes every upsert; replayed IDs overwrite rather than add,
	// so the target count must never exceed applied and must equal the number
	// of distinct IDs we believe we wrote.
	if targetCount > expectedDistinct {
		return newMergeError(MergePhaseVerify, "doc count reconciliation",
			fmt.Errorf("target doc count %d exceeds applied distinct count %d", targetCount, expectedDistinct))
	}

	partial := false
	if status.SkipRatio > j.cfg.MaxSkipRatio {
		partial = true
	}
	return j.state.mutate(func(s *MergeStatus) {
		s.PartialSuccess = partial
	})
}

// aliasLayoutMatchesView reports whether the alias' current members are
// exactly the same Index instances, in the same order, as the members the
// consolidation view was acquired over.  Identity (pointer) comparison is used
// because names are user-assignable and need not be unique.
func (j *mergeJob) aliasLayoutMatchesView() bool {
	if j.view == nil {
		return false
	}
	j.alias.mutex.RLock()
	current := j.alias.indexes
	if len(current) != len(j.view.members) {
		j.alias.mutex.RUnlock()
		return false
	}
	for ordinal, member := range j.view.members {
		if current[ordinal] != member.idx {
			j.alias.mutex.RUnlock()
			return false
		}
	}
	j.alias.mutex.RUnlock()
	return true
}

// switchAlias performs the topology change atomically under the alias write
// lock.  Readers either see the full pre-switch member set or the full
// post-switch member set, never a mix; and the target is never visible before
// this point.
func (j *mergeJob) switchAlias() error {
	j.switchMu.Lock()
	defer j.switchMu.Unlock()

	if j.isCancelled() {
		return newMergeError(MergePhaseSwitch, "cancellation", errMergeCancelled)
	}

	if err := j.state.mutate(func(s *MergeStatus) {
		s.State = stateSwitching
		s.Phase = MergePhaseSwitch
		s.OpenViews = 1
	}); err != nil {
		return err
	}

	start := time.Now()

	// Guard against a topology change (concurrent Add/Remove/Swap) during the
	// run.  The view pinned the original members; if the alias no longer holds
	// exactly those members in the same order, swapping now could remove a
	// member this consolidation never saw or leave an unexpected member in
	// place.  Refuse to touch the alias and keep the pre-switch view serving.
	if !j.aliasLayoutMatchesView() {
		return newMergeError(MergePhaseSwitch, "topology check",
			fmt.Errorf("%w: alias members changed during consolidation", ErrorMergeTargetIncompatible))
	}

	// Snapshot the source members currently registered (the view still pins
	// their data).  Swap replaces them with the single consolidated target in
	// one critical section.
	j.alias.mutex.RLock()
	sourceIndexes := append([]Index(nil), j.alias.indexes...)
	j.alias.mutex.RUnlock()

	// The actual swap is one locked operation on the alias; concurrent
	// searches block on the RWMutex and observe either the old slice or the
	// new one wholesale.
	j.alias.Swap([]Index{j.target}, sourceIndexes)

	switchDuration := time.Since(start)

	// The view can now be released: no reader depends on the old members via
	// the job anymore.  Engine-level reclamation of old segments is unblocked,
	// but the old indexes themselves are not deleted (per hard constraint);
	// ownership/lifecycle stays with whoever opened them.
	releaseErr := j.view.Release()
	j.view = nil

	return j.state.mutate(func(s *MergeStatus) {
		s.State = stateCompleted
		s.OpenViews = 0
		s.SwitchDurationNS = switchDuration.Nanoseconds()
		s.FinishedAt = nowRFC3339()
		s.Pending = 0
		s.CatchupPending = 0
		s.reconcilePending()
		if releaseErr != nil {
			s.FailureReason = "view released with error after switch: " + releaseErr.Error()
		}
	})
}

// recordFailure marks the job failed without touching the alias, so the
// pre-switch members keep serving.  The half-built target stays invisible to
// queries; owned targets are removed by cleanupOnExit.
func (j *mergeJob) recordFailure(phase MergePhase, cause error) {
	_ = j.state.mutate(func(s *MergeStatus) {
		s.State = stateFailed
		s.Phase = phase
		s.FailurePhase = phase
		s.FailureReason = cause.Error()
		s.FinishedAt = nowRFC3339()
	})
}

// cleanupOnExit releases the view and, only when the failure left a target
// that cannot be resumed, closes and removes an owned target.  A failure or
// cancellation before the switch is resumable from the persisted cursor, so
// the half-built target is deliberately kept on disk and the state is left as
// failed rather than discarded.  Source indexes are never removed.
func (j *mergeJob) cleanupOnExit() {
	if j.view != nil {
		_ = j.view.Release()
		j.view = nil
	}
	state := j.getState()
	if state == nil {
		return
	}
	status := state.snapshot()
	if !j.targetOwned || j.target == nil || status.State == stateCompleted {
		return
	}
	// Keep the target open for resume: it is a half-built but internally
	// consistent index (batches are atomic), and it is not queryable through
	// the alias because the switch never happened.  Closing releases file
	// handles; the directory is retained so ResumeConsolidation can Open it.
	_ = j.target.Close()
}

// reconcilePending keeps Pending consistent with the authoritative counters.
// Pending is the number of view entries still awaiting application: the
// enumerated total minus what has been applied, conflicted away or skipped.
// Catch-up work is tracked separately in CatchupApplied/CatchupPending and
// must not be double counted here.
func (s *MergeStatus) reconcilePending() {
	resolved := s.Applied + s.Conflicts + s.Skipped
	// Applied includes catch-up writes, which are not part of TotalEnumerated;
	// subtract them so the main-view accounting stays exact.
	mainApplied := s.Applied
	if s.CatchupApplied <= mainApplied {
		mainApplied -= s.CatchupApplied
	}
	mainResolved := mainApplied + s.Conflicts + s.Skipped
	if mainResolved >= s.TotalEnumerated {
		s.Pending = 0
	} else {
		s.Pending = s.TotalEnumerated - mainResolved
	}
	_ = resolved
}

// recomputeSkipRatio derives the ratio from the same authoritative counters.
func (s *MergeStatus) recomputeSkipRatio() {
	observed := s.Applied + s.Skipped
	if observed == 0 {
		s.SkipRatio = 0
		return
	}
	s.SkipRatio = float64(s.Skipped) / float64(observed)
}
