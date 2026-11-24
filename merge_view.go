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
	"strconv"
	"strings"
	"sync"

	index "github.com/blevesearch/bleve_index_api"
)

// ViewMember describes one fixed member of an IndexView: the bleve Index as
// it was at view acquisition time, together with the point-in-time low-level
// reader that all view reads go through.
type ViewMember struct {
	// Name is Index.Name() captured at view acquisition time.
	Name string
	// Ordinal is the deterministic position of this member inside the alias,
	// zero based.  It is the primary component of the merge ordering.
	Ordinal int

	idx    Index
	reader index.IndexReader
}

// IndexView is a read-only, point-in-time view over the members of an index
// alias.  Both the member set and each member's contents are frozen at
// acquisition time: writes that land afterwards are never visible through the
// view.
//
// A view must be released after use.  Until Release returns, the underlying
// engine keeps every segment or key-value snapshot the view depends on alive,
// even if concurrent merges or cleanup would otherwise reclaim them; resource
// reclamation is simply deferred (isolation takes precedence over space
// reclamation).
//
// IndexView is safe for concurrent use by multiple goroutines.
type IndexView struct {
	alias   *indexAliasImpl
	members []*ViewMember

	mu     sync.Mutex
	closed bool
}

// AcquireView returns a point-in-time read-only view over the alias' current
// members.  The member list, their order and their on-disk state are fixed by
// this call; subsequent Add/Remove/Swap and subsequent writes do not affect the
// view.
//
// The returned view must be released with Release.  It is obtained through the
// public alias entry point and only uses public index APIs.
func (i *indexAliasImpl) AcquireView() (*IndexView, error) {
	i.mutex.RLock()
	defer i.mutex.RUnlock()

	if !i.open {
		return nil, ErrorIndexClosed
	}
	if len(i.indexes) < 1 {
		return nil, ErrorAliasEmpty
	}

	// Snapshot the member slice first.  All readers below are opened while
	// holding the alias read lock, so no topology change can interleave with
	// the acquisition sequence.
	members := make([]*ViewMember, len(i.indexes))
	for ordinal, idx := range i.indexes {
		members[ordinal] = &ViewMember{
			Name:    idx.Name(),
			Ordinal: ordinal,
			idx:     idx,
		}
	}

	rollback := func(upto int) {
		for k := 0; k < upto; k++ {
			_ = members[k].reader.Close()
		}
	}

	for ordinal, idx := range i.indexes {
		advanced, err := idx.Advanced()
		if err != nil {
			rollback(ordinal)
			return nil, newMergeError(MergePhasePrepare, "advanced index", err)
		}
		// Reader() returns a ref-counted point-in-time snapshot (scorch) or a
		// snapshot KV reader (upsidedown/gtreap).  Holding it open is exactly
		// what pins the view's files and handles against concurrent cleanup.
		reader, err := advanced.Reader()
		if err != nil {
			rollback(ordinal)
			return nil, newMergeError(MergePhasePrepare, "open snapshot reader", err)
		}
		members[ordinal].reader = reader
	}

	return &IndexView{
		alias:   i,
		members: members,
	}, nil
}

// Members returns the fixed member descriptors of the view, in deterministic
// alias order.  The returned slice must not be modified.
func (v *IndexView) Members() []*ViewMember {
	return v.members
}

// DocCount returns the total number of live documents visible in the view,
// summed from every frozen member snapshot.
func (v *IndexView) DocCount() (uint64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return 0, ErrorViewClosed
	}

	var total uint64
	for _, m := range v.members {
		count, err := m.reader.DocCount()
		if err != nil {
			return total, err
		}
		total += count
	}
	return total, nil
}

// EnumeratedCount returns the exact number of entries a full
// DocumentBatch walk would produce, counting a cross-member duplicate ID once
// per member that contains it.  It is the authoritative denominator for merge
// accounting (applied + conflicts + skipped).
func (v *IndexView) EnumeratedCount() (uint64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return 0, ErrorViewClosed
	}

	var total uint64
	for _, member := range v.members {
		docIDReader, err := member.reader.DocIDReaderAll()
		if err != nil {
			return total, err
		}
		for {
			internalID, err := docIDReader.Next()
			if err != nil {
				_ = docIDReader.Close()
				return total, err
			}
			if internalID == nil {
				break
			}
			total++
		}
		if err := docIDReader.Close(); err != nil {
			return total, err
		}
	}
	return total, nil
}

// ViewDocument is one enumerated view entry.  Doc is the reconstructed stored
// document and SourceMember is the ordinal (ViewMember.Ordinal) it came from.
type ViewDocument struct {
	Doc          index.Document
	SourceMember int
}

// ViewDocBatch is a deterministic batch of documents pulled from a view.
type ViewDocBatch struct {
	Docs []ViewDocument
	// Cursor is the opaque, ordered resumption token.  Pass it unchanged to
	// the next DocumentBatch call; it is empty when the whole view has been
	// enumerated.
	Cursor string
	// Done is true once every live document in the view has been returned.
	Done bool
	// SkippedIDs lists document IDs whose existence the snapshot confirmed but
	// whose stored fields could not be reconstructed.  The merge driver counts
	// these as skipped documents; the IDs remain part of the view's DocCount.
	SkippedIDs []string
}

// DocumentBatch enumerates at most batchSize live documents from the view,
// starting at cursor (empty cursor means "from the beginning").
//
// Enumeration order is fully deterministic: members are visited by ordinal,
// and within a member documents follow the engine's internal identifier order
// (DocIDReader order).  The same view always yields the same sequence,
// independent of map iteration or goroutine scheduling.
//
// Documents are streamed in batches so views with tens of millions of
// documents never need to materialize in memory at once.
func (v *IndexView) DocumentBatch(cursor string, batchSize int) (*ViewDocBatch, error) {
	if batchSize <= 0 {
		batchSize = 256
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil, ErrorViewClosed
	}

	startMember, resumeID, err := parseViewCursor(cursor)
	if err != nil {
		return nil, err
	}
	if startMember > len(v.members) {
		return nil, fmt.Errorf("merge: cursor member ordinal %d out of range (view has %d members)",
			startMember, len(v.members))
	}

	rv := &ViewDocBatch{
		Docs: make([]ViewDocument, 0, batchSize),
	}

	// nextMember is the member to resume from; resumeID, when non-empty, is
	// the external ID of the last document already delivered in that member.
	// Reopening a reader is cheap on both engines (a ref-counted snapshot or a
	// persistent KV snapshot), and scanning forward past the resume ID makes
	// progress independent of how internal IDs map to external IDs.
	nextMember, resumeExternalID := startMember, resumeID

	for ordinal := nextMember; ordinal < len(v.members) && len(rv.Docs) < batchSize; ordinal++ {
		member := v.members[ordinal]
		docIDReader, err := member.reader.DocIDReaderAll()
		if err != nil {
			return nil, err
		}

		// Resume semantics are anchored on the external document ID, which is
		// stable across snapshots and processes (unlike engine internal doc
		// numbers, which shift when segments are added).
		//
		// Two resume modes:
		//   - advanceHit: the external ID resolves in this snapshot, so
		//     Advance positions the iterator at it and exactly that entry is
		//     skipped once.
		//   - scanFallback: the boundary doc cannot be resolved; iterate from
		//     the member start, dropping every entry whose external ID is
		//     lexicographically <= the resume ID, then resume normally.
		var skipExact string // external ID to skip exactly once (advance mode)
		var scanUntil string // resume ID to scan past (fallback mode)
		if ordinal == nextMember && resumeExternalID != "" {
			internalID, idErr := member.reader.InternalID(resumeExternalID)
			if idErr == nil && internalID != nil {
				if _, aErr := docIDReader.Advance(internalID); aErr != nil {
					_ = docIDReader.Close()
					return nil, aErr
				}
				skipExact = resumeExternalID
			} else {
				scanUntil = resumeExternalID
			}
		}

		lastExternalID := ""
		for len(rv.Docs) < batchSize {
			internalID, err := docIDReader.Next()
			if err != nil {
				_ = docIDReader.Close()
				return nil, err
			}
			if internalID == nil {
				break // this member is exhausted
			}
			externalID, err := member.reader.ExternalID(internalID)
			if err != nil {
				_ = docIDReader.Close()
				return nil, err
			}
			if skipExact != "" {
				// Advance placed us at the boundary; drop it once and proceed.
				if externalID == skipExact {
					skipExact = ""
					continue
				}
				// Boundary vanished between InternalID and Next: the current
				// entry is already its successor, so stop skipping.
				skipExact = ""
			}
			if scanUntil != "" {
				// Fallback mode: drop entries until the exact resume boundary
				// is consumed.  External IDs have no defined order relative to
				// the iterator, so matching is by identity, not comparison.
				// If the boundary was deleted it is never seen and the member
				// is fully scanned (a rare, correctness-first path).
				if externalID == scanUntil {
					scanUntil = ""
				}
				continue
			}
			doc, err := member.reader.Document(externalID)
			if err != nil {
				_ = docIDReader.Close()
				return nil, err
			}
			if doc == nil {
				// The snapshot lists the ID but stored fields are unavailable
				// (index read inconsistency).  Record the ID so the merge
				// driver can count it as skipped rather than losing it.
				rv.SkippedIDs = append(rv.SkippedIDs, externalID)
				lastExternalID = externalID
				continue
			}
			rv.Docs = append(rv.Docs, ViewDocument{
				Doc:          doc,
				SourceMember: ordinal,
			})
			lastExternalID = externalID
		}
		if cerr := docIDReader.Close(); cerr != nil {
			return nil, cerr
		}

		if len(rv.Docs) >= batchSize {
			// Batch boundary landed inside this member: resume here next time.
			rv.Cursor = encodeViewCursor(ordinal, lastExternalID)
			rv.Done = false
			return rv, nil
		}
		// Member fully drained; the next member starts from its beginning.
		resumeExternalID = ""
	}

	rv.Done = true
	rv.Cursor = ""
	return rv, nil
}

// Release closes every frozen snapshot, decrementing the engine-level
// references that pinned segments and files.  After Release the view and its
// members must not be used.  Releasing twice is a no-op.
func (v *IndexView) Release() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return nil
	}
	v.closed = true

	var firstErr error
	for _, m := range v.members {
		if m.reader != nil {
			if err := m.reader.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

// Cursor encoding: "<ordinal>:<quoted external id>".  The external document ID
// is the resume anchor because it stays stable across snapshots and process
// restarts; it is Go-quoted so any rune (including the ':' separator)
// round-trips safely.  An empty cursor means "start of the view"; a trailing
// empty id field means "start of that member".
func encodeViewCursor(ordinal int, externalID string) string {
	if externalID == "" {
		return strconv.Itoa(ordinal) + ":"
	}
	return strconv.Itoa(ordinal) + ":" + strconv.Quote(externalID)
}

func parseViewCursor(cursor string) (ordinal int, externalID string, err error) {
	if cursor == "" {
		return 0, "", nil
	}
	colon := strings.IndexByte(cursor, ':')
	if colon < 0 {
		return 0, "", fmt.Errorf("%w: missing separator in cursor %q", ErrorMergeStateCorrupt, cursor)
	}
	ordinal, err = strconv.Atoi(cursor[:colon])
	if err != nil {
		return 0, "", fmt.Errorf("%w: bad ordinal in cursor %q: %v", ErrorMergeStateCorrupt, cursor, err)
	}
	quoted := cursor[colon+1:]
	if quoted == "" {
		return ordinal, "", nil
	}
	unquoted, qerr := strconv.Unquote(quoted)
	if qerr != nil {
		return 0, "", fmt.Errorf("%w: bad external id in cursor %q: %v", ErrorMergeStateCorrupt, cursor, qerr)
	}
	return ordinal, unquoted, nil
}
