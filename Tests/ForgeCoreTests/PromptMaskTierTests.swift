import Testing
import MLXToolKit
@testable import ForgeCore

/// AB-A-0105: SAM 2.1-S joins EdgeTAM as a second `promptSegment` registration — same capability, its own
/// `ForgeModel` case, its own display name, its own PackageID (explicit id). Pinned so the two never collapse
/// onto one registration the way the three Restormer variants once did (AB-L-0160).
@Suite struct PromptMaskTierTests {
    @Test func qualityTierIsASecondPromptSegmentRegistration() {
        #expect(ForgeModel.promptMask.capability == .promptSegment)
        #expect(ForgeModel.promptMaskQuality.capability == .promptSegment)
        #expect(ForgeModel.promptMaskQuality.modelName == "SAM 2.1")
        #expect(ForgeModel.promptMask.modelName == "EdgeTAM")
        #expect(ForgeModel.promptMaskQuality.rawValue != ForgeModel.promptMask.rawValue)
        #expect(ForgeModel.allCases.contains(.promptMaskQuality))
    }

    /// Exactly two prompt-mask backers, and both are reachable from the case list a host iterates.
    @Test func promptSegmentHasExactlyTwoBackers() {
        let backers = ForgeModel.allCases.filter { $0.capability == .promptSegment }
        #expect(Set(backers) == Set([.promptMask, .promptMaskQuality]))
    }
}
