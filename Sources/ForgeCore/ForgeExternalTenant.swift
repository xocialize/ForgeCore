//
//  ForgeExternalTenant.swift
//  ForgeCore
//
//  An app-side GPU tenant — the Forge canvas first — declared to the engine the app already owns, so admission counts its
//  bytes and R-MEM-1 can ask it to shrink (mlx-engine-swift ≥ 0.61.0, contract 1.46.0, AB-A-0097). ForgeCore holds the
//  ENGINE side only: what the tenant sheds, and why, is the tenant's policy (the canvas's lives in ForgeCanvasUI as
//  `CanvasPressurePolicy`, engine-free), handed in as a closure. That keeps the heavy pin here (B owns it) and the policy
//  with the thing it describes — an app composes the two and never pins the engine itself.
//
import Foundation
import os
import MLXServeCore

public final class ForgeExternalTenant: @unchecked Sendable {
    /// What the tenant holds: bytes it keeps between frames, bytes it borrows per frame, and how many of the persistent
    /// bytes it could give back during a model RUN without having to re-create them on its next frame.
    public struct Footprint: Sendable, Equatable {
        public var persistentBytes: Int
        public var transientBytes: Int
        public var sheddableOnRun: Int
        public init(persistentBytes: Int, transientBytes: Int, sheddableOnRun: Int) {
            self.persistentBytes = persistentBytes; self.transientBytes = transientBytes; self.sheddableOnRun = sheddableOnRun
        }
    }

    /// Why the engine asks (contract 1.46.0).
    public enum Reason: String, Sendable {
        /// A package is loading, or a resident one no longer fits its reserve: shedding caches is worth it.
        case admission
        /// A resident package runs while the process is over R-MEM-1's ceiling: shed only what won't be needed next frame.
        case run
    }

    /// One answered request, for receipts.
    public struct Shrink: Sendable {
        public let requested: UInt64
        public let shed: UInt64
        public let reason: Reason
        public let milliseconds: Double
        public let note: String
    }

    /// Sheds bytes for a request; returns what it shed and a note (the rungs it took). Called on the engine's thread; it
    /// must answer well inside the engine's 2 s and leave the declaration current (call `update`).
    public typealias Shedder = @Sendable (_ requested: UInt64, _ reason: Reason) -> (shed: UInt64, note: String)

    public let id: String
    private var handle: ExternalTenant?
    private let current = OSAllocatedUnfairLock(initialState: Footprint(persistentBytes: 0, transientBytes: 0, sheddableOnRun: 0))
    private let log = OSAllocatedUnfairLock(initialState: (shrinks: [Shrink](), emptyAnswers: 0))

    public init(id: String) { self.id = id }

    public var footprint: Footprint { current.withLock { $0 } }
    public var shrinks: [Shrink] { log.withLock { $0.shrinks } }
    /// Requests answered 0 at once because nothing was left to give for that reason (no wait on a frame).
    public var emptyAnswers: Int { log.withLock { $0.emptyAnswers } }

    /// Declares the tenant to `service`'s engine (replacing an earlier declaration under the same id).
    public func attach(to service: ForgeEngineService, footprint: Footprint, shedder: @escaping Shedder) async {
        current.withLock { $0 = footprint }
        let tenant = await service.engine.registerExternalTenant(
            id: id, persistentBytes: UInt64(footprint.persistentBytes), transientBytes: UInt64(footprint.transientBytes),
            onShrinkRequest: { [weak self] (request: ExternalShrinkRequest) in
                self?.answer(request, shedder) ?? 0
            })
        handle?.withdraw()
        handle = tenant
    }

    /// Re-declares — call on every footprint change. Cheap; safe from any thread.
    public func update(_ footprint: Footprint) {
        current.withLock { $0 = footprint }
        handle?.update(persistentBytes: UInt64(footprint.persistentBytes), transientBytes: UInt64(footprint.transientBytes))
    }

    public func detach() { handle?.withdraw(); handle = nil }

    private func answer(_ request: ExternalShrinkRequest, _ shedder: Shedder) -> UInt64 {
        let reason: Reason
        switch request.reason { case .runUnderRealPressure: reason = .run; default: reason = .admission }
        let fp = footprint
        // Nothing left to give for this reason: answer at once, without waiting on the tenant (AB-A-0093).
        if (reason == .run ? fp.sheddableOnRun : fp.persistentBytes) == 0 {
            log.withLock { $0.emptyAnswers += 1 }
            return 0
        }
        let t0 = Date()
        let (shed, note) = shedder(request.requestedBytes, reason)
        log.withLock {
            $0.shrinks.append(Shrink(requested: request.requestedBytes, shed: shed, reason: reason,
                                     milliseconds: Date().timeIntervalSince(t0) * 1000, note: note))
        }
        return shed
    }
}
