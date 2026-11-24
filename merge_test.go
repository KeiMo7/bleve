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
	"sort"
	"testing"

	"github.com/blevesearch/bleve/v2/mapping"
	index "github.com/blevesearch/bleve_index_api"
)

// mergeTestEnv bundles the per-test directories and indexes used by the
// consolidation tests.  cleanup closes everything and removes temp dirs.
type mergeTestEnv struct {
	t       *testing.T
	rootDir string
	disk    bool // false => memory-only indexes

	mapping mapping.IndexMapping
	alias   *indexAliasImpl
	members []Index
	paths   []string
}

func newMergeTestEnv(t *testing.T, disk bool, memberCount int) *mergeTestEnv {
	t.Helper()
	env := &mergeTestEnv{
		t:       t,
		disk:    disk,
		mapping: NewIndexMapping(),
	}
	// Consolidation reconstructs documents from stored fields (the engine's
	// Document() API cannot rebuild non-stored/dynamic-only fields), so the
	// test indexes enable dynamic storage.  This mirrors the real-world
	// precondition for index consolidation.
	env.mapping.(*mapping.IndexMappingImpl).StoreDynamic = true
	if disk {
		env.rootDir = t.TempDir()
	}
	for i := 0; i < memberCount; i++ {
		var idx Index
		var err error
		if disk {
			path := filepath.Join(env.rootDir, fmt.Sprintf("member-%d", i))
			env.paths = append(env.paths, path)
			idx, err = New(path, env.mapping)
		} else {
			idx, err = NewMemOnly(env.mapping)
		}
		if err != nil {
			t.Fatalf("create member %d: %v", i, err)
		}
		idx.SetName(fmt.Sprintf("member-%d", i))
		env.members = append(env.members, idx)
	}
	// Hand the alias an independent copy of the member slice: Add/Remove/Swap
	// mutate the alias' backing array in place, so sharing env.members would
	// let a swap overwrite the test's own member tracking.
	aliasMembers := append([]Index(nil), env.members...)
	env.alias = NewIndexAlias(aliasMembers...)
	env.alias.SetName("test-alias")
	return env
}

func (e *mergeTestEnv) targetPath(name string) string {
	if !e.disk {
		return ""
	}
	return filepath.Join(e.rootDir, name)
}

func (e *mergeTestEnv) stateDir() string {
	if !e.disk {
		return ""
	}
	return e.rootDir
}

func (e *mergeTestEnv) cleanup() {
	for _, idx := range e.members {
		_ = idx.Close()
	}
	if e.rootDir != "" {
		_ = os.RemoveAll(e.rootDir)
	}
}

// indexDocs writes a deterministic set of docs into member ordinal.
func (e *mergeTestEnv) indexDocs(ordinal int, docs map[string]string) {
	e.t.Helper()
	idx := e.members[ordinal]
	batch := idx.NewBatch()
	ids := make([]string, 0, len(docs))
	for id := range docs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if err := batch.Index(id, map[string]interface{}{"name": docs[id]}); err != nil {
			e.t.Fatalf("batch index %s: %v", id, err)
		}
	}
	if err := idx.Batch(batch); err != nil {
		e.t.Fatalf("commit batch on member %d: %v", ordinal, err)
	}
}

// runBackends executes a test against both disk (scorch) and memory
// (upsidedown/gtreap) indexes, as required by the "both index kinds"
// constraint.
func runBackends(t *testing.T, fn func(t *testing.T, env *mergeTestEnv)) {
	for _, disk := range []bool{false, true} {
		kind := "mem"
		if disk {
			kind = "disk"
		}
		t.Run(kind, func(t *testing.T) {
			env := newMergeTestEnv(t, disk, 3)
			defer env.cleanup()
			fn(t, env)
		})
	}
}

// TestMergeBasicConsolidation covers forms 1-4, 7 and the counting contract:
// a frozen view, batched copy, deterministic conflict resolution and an atomic
// switch that leaves the alias pointing at exactly the consolidated index.
func TestMergeBasicConsolidation(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		// Member 0: a-1..a-40; member 1: b-1..b-30 plus a duplicate id
		// "shared" with different content; member 2: c-1..c-20 and another
		// "shared" (three-way conflict) plus a duplicate of a-1.
		docs0 := map[string]string{}
		for i := 0; i < 40; i++ {
			docs0[fmt.Sprintf("a-%d", i)] = fmt.Sprintf("alpha %d", i)
		}
		docs0["shared"] = "from-member-0"
		docs0["a-1"] = "alpha-1-original"
		env.indexDocs(0, docs0)

		docs1 := map[string]string{"shared": "from-member-1"}
		for i := 0; i < 30; i++ {
			docs1[fmt.Sprintf("b-%d", i)] = fmt.Sprintf("bravo %d", i)
		}
		env.indexDocs(1, docs1)

		docs2 := map[string]string{"shared": "from-member-2", "a-1": "alpha-1-clobbered"}
		for i := 0; i < 20; i++ {
			docs2[fmt.Sprintf("c-%d", i)] = fmt.Sprintf("charlie %d", i)
		}
		env.indexDocs(2, docs2)

		// Form 1 & 2: acquire a view and confirm it is frozen.
		view, err := env.alias.AcquireView()
		if err != nil {
			t.Fatalf("AcquireView: %v", err)
		}
		if len(view.Members()) != 3 {
			t.Fatalf("expected 3 view members, got %d", len(view.Members()))
		}
		viewCount, err := view.DocCount()
		if err != nil {
			t.Fatalf("view DocCount: %v", err)
		}
		// Per-member live counts: 41 + 31 + 22 = 94.  DocCount is distinct
		// within each member, so the intra-member rewrite of a-1 counts once.
		if viewCount != 94 {
			t.Fatalf("expected 94 entries in view, got %d", viewCount)
		}

		// A write after view acquisition must not appear in the view.
		env.indexDocs(0, map[string]string{"late-doc": "invisible"})
		batch, err := view.DocumentBatch("", 16)
		if err != nil {
			t.Fatalf("DocumentBatch: %v", err)
		}
		for _, d := range batch.Docs {
			if d.Doc.ID() == "late-doc" {
				t.Fatalf("post-acquisition write leaked into frozen view")
			}
		}
		if err := view.Release(); err != nil {
			t.Fatalf("Release: %v", err)
		}
		// Double release is a no-op.
		if err := view.Release(); err != nil {
			t.Fatalf("second Release: %v", err)
		}

		// Run the consolidation with a small batch size to exercise many
		// batches and the cursor path.
		cfg := MergeConfig{
			JobID:      "job-basic",
			BatchSize:  7,
			TargetPath: env.targetPath("target"),
			StateDir:   env.stateDir(),
		}
		result, err := env.alias.StartConsolidation(cfg)
		if err != nil {
			t.Fatalf("StartConsolidation: %v", err)
		}
		if !result.Swapped {
			t.Fatal("expected successful swap")
		}

		status := result.Status
		// Deterministic conflict rule: "a-1" loses once (member 2 vs member 0)
		// and "shared" loses twice (members 1 and 2 vs member 0), so there are
		// exactly 3 conflict occurrences and 2 conflicted IDs.
		if status.Conflicts != 3 {
			t.Fatalf("expected 3 conflict occurrences (a-1 x1, shared x2), got %d", status.Conflicts)
		}
		if len(status.ConflictSamples) != 2 {
			t.Fatalf("expected 2 distinct conflict sample IDs, got %d", len(status.ConflictSamples))
		}
		// Deterministic winner: every conflict sample names member 0.
		for _, sample := range status.ConflictSamples {
			if sample.WinnerMember != 0 || sample.WinnerMemberName != "member-0" {
				t.Fatalf("conflict %q resolved by member %d (%s), want member-0",
					sample.DocID, sample.WinnerMember, sample.WinnerMemberName)
			}
		}
		// Accounting identity over the enumerated entries: every entry the
		// view enumerates is applied, conflicted or skipped exactly once.
		enumerated := status.Applied + status.Conflicts + status.Skipped
		if enumerated != status.TotalEnumerated {
			t.Fatalf("accounting mismatch: applied %d + conflicts %d + skipped %d = %d, want %d",
				status.Applied, status.Conflicts, status.Skipped, enumerated, status.TotalEnumerated)
		}
		if status.Applied != 92 {
			t.Fatalf("expected 92 applied docs, got %d", status.Applied)
		}
		if status.BatchesCommitted == 0 {
			t.Fatal("expected committed batches to be recorded")
		}
		if status.Pending != 0 {
			t.Fatalf("expected zero pending after switch, got %d", status.Pending)
		}
		if status.State != stateCompleted {
			t.Fatalf("expected completed state, got %s", status.State)
		}
		if status.SwitchDurationNS <= 0 {
			t.Fatal("expected switch duration to be measured")
		}

		// Form 7: the alias now has exactly one member, the target.
		env.alias.mutex.RLock()
		memberCount := len(env.alias.indexes)
		onlyMember := env.alias.indexes[0]
		env.alias.mutex.RUnlock()
		if memberCount != 1 || onlyMember != result.Target {
			t.Fatalf("alias not atomically swapped to target: members=%d", memberCount)
		}

		// Deterministic conflict winner: lowest ordinal wins.  The target
		// holds one document per unique ID, i.e. every enumerated entry minus
		// the losing conflict occurrences.
		expectedUnique := status.TotalEnumerated - status.Conflicts - status.Skipped
		targetCount, err := result.Target.DocCount()
		if err != nil {
			t.Fatalf("target DocCount: %v", err)
		}
		if targetCount != expectedUnique {
			t.Fatalf("expected target count %d, got %d", expectedUnique, targetCount)
		}
		doc, err := result.Target.Document("shared")
		if err != nil || doc == nil {
			t.Fatalf("load shared: doc=%v err=%v", doc, err)
		}
		foundWinner := false
		doc.VisitFields(func(f index.Field) {
			if f.Name() == "name" && string(f.Value()) == "from-member-0" {
				foundWinner = true
			}
		})
		if !foundWinner {
			t.Fatal("conflict winner was not the lowest-ordinal member")
		}

		// Search through the alias still works end to end and returns the
		// consolidated data.
		query := NewMatchQuery("alpha")
		searchReq := NewSearchRequest(query)
		searchReq.Size = 100
		searchResult, err := env.alias.Search(searchReq)
		if err != nil {
			t.Fatalf("post-switch search: %v", err)
		}
		if searchResult.Total != 40 {
			t.Fatalf("expected 40 alpha matches, got %d", searchResult.Total)
		}
	})
}

// TestMergeSingleFlight verifies form 9's mutual exclusion: a second
// consolidation on the same alias is explicitly rejected.
func TestMergeSingleFlight(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		env.indexDocs(0, map[string]string{"x-1": "one"})
		cfg := MergeConfig{JobID: "job-1", TargetPath: env.targetPath("t1"), StateDir: env.stateDir()}

		// Hold a fake running job in the registry by starting one in a
		// goroutine and waiting until it registers; consolidation is fast, so
		// instead validate the guard directly by occupying the slot.
		globalMergeMu.Lock()
		placeholder := &mergeJob{cfg: MergeConfig{JobID: "placeholder"}}
		runningMergeJobs[env.alias] = placeholder
		globalMergeMu.Unlock()

		_, err := env.alias.StartConsolidation(cfg)
		if err != ErrorMergeAlreadyRunning {
			t.Fatalf("expected ErrorMergeAlreadyRunning, got %v", err)
		}

		globalMergeMu.Lock()
		delete(runningMergeJobs, env.alias)
		globalMergeMu.Unlock()
	})
}

// TestMergeViewDeterministicOrder verifies form 3/requirement 2 (determinism):
// repeated enumeration of the same view yields the identical ID sequence, and
// resuming from every intermediate cursor reconstructs the full sequence.
func TestMergeViewDeterministicOrder(t *testing.T) {
	runBackends(t, func(t *testing.T, env *mergeTestEnv) {
		docs := map[string]string{}
		for i := 0; i < 50; i++ {
			docs[fmt.Sprintf("doc-%03d", i)] = fmt.Sprintf("v%d", i)
		}
		env.indexDocs(1, docs)

		view, err := env.alias.AcquireView()
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = view.Release() }()

		collect := func(batchSize int) []string {
			var ids []string
			cursor := ""
			for {
				batch, err := view.DocumentBatch(cursor, batchSize)
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
			return ids
		}

		seq1 := collect(5)
		seq2 := collect(13)
		seq3 := collect(1000)
		if len(seq1) != 50 || len(seq2) != 50 || len(seq3) != 50 {
			t.Fatalf("unexpected lengths %d %d %d", len(seq1), len(seq2), len(seq3))
		}
		for i := range seq1 {
			if seq1[i] != seq2[i] || seq2[i] != seq3[i] {
				t.Fatalf("non-deterministic enumeration at %d: %s %s %s", i, seq1[i], seq2[i], seq3[i])
			}
		}

		// Resume from every cursor emitted with batch size 5 and verify the
		// tail matches the full sequence.
		cursor := ""
		delivered := 0
		for {
			batch, err := view.DocumentBatch(cursor, 5)
			if err != nil {
				t.Fatal(err)
			}
			delivered += len(batch.Docs)
			if batch.Done {
				break
			}
			tail, err := view.DocumentBatch(batch.Cursor, 5)
			if err != nil {
				t.Fatal(err)
			}
			for k, d := range tail.Docs {
				if d.Doc.ID() != seq1[delivered+k] {
					t.Fatalf("cursor resume mismatch at %d", delivered+k)
				}
			}
			cursor = batch.Cursor
		}
	})
}
