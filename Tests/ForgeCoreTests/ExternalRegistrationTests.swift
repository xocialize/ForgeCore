//
//  ExternalRegistrationTests.swift — ForgeCore
//
//  The injection seam's contract. The live success path (the CoreAI package registering through
//  it) is proven by a scratch driver against the real engine — here we pin what the seam promises
//  structurally: outcomes are recorded per injection, failures are facts not crashes, and lookup
//  by name answers exactly what was injected.
//

import Foundation
import MLXToolKit
import Testing

@testable import ForgeCore

@Suite("ExternalRegistration seam")
@MainActor
struct ExternalRegistrationTests {

    struct TestError: Error {}

    @Test func failingRegistrationIsAFactNotACrash() async {
        let service = ForgeEngineService(externalRegistrations: [
            ExternalRegistration(name: "coreai-upscale", capability: .imageUpscale) { _ in
                throw TestError()
            }
        ])
        // Built-ins will also run; we only assert the external half here.
        await service.registerAll()
        #expect(service.externalOutcomes.count == 1)
        let outcome = service.externalOutcomes[0]
        #expect(outcome.name == "coreai-upscale")
        #expect(outcome.capability == .imageUpscale)
        #expect(!outcome.isRegistered)
        #expect(outcome.failure != nil)
        #expect(service.packageID(forExternal: "coreai-upscale") == nil)
        #expect(service.packageID(forExternal: "never-injected") == nil)
    }

    @Test func noInjectionsMeansEmptyOutcomes() {
        let service = ForgeEngineService()
        #expect(service.externalRegistrations.isEmpty)
        #expect(service.externalOutcomes.isEmpty)
    }
}
