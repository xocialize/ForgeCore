//
//  NoReferenceQuality.swift — ForgeCore
//
//  The **second oracle**, and the reason it now exists here.
//
//  🚨 `CLAUDE.md` states the rule: *"Two oracles by design: SigLIP2 NR-IQA gates whether to restore
//  (no reference needed on a degraded source); SSIMULACRA2 gates how hard to compress
//  (full-reference)."* ForgeCore shipped only the second — `MediaMeasure`, via media-bridge — so every
//  quality question a consumer could ask it was a *fidelity* question. V10 then measured what that
//  costs, and the number is not subtle: on `Bayer_photo` 512→2048, Lanczos (SSIM 0.8654 / MAE 3.58)
//  beat Real-ESRGAN (0.8362 / 4.48) and SeedVR2-3B (0.6276 / 11.77), while the **detail** ordering was
//  exactly inverted — SeedVR2 2.98 > Real-ESRGAN 2.24 > Lanczos 2.05 against a reference of 3.72.
//
//  🔑 **That is not a bug in either metric; it is what "full-reference" means.** A distance to the
//  reference charges a *generative* upscaler for every plausible-but-different pixel it invents, and
//  charges an interpolator nothing for inventing none — the method that adds no information above the
//  source Nyquist is the method that cannot be wrong about it. So a full-reference metric is
//  structurally biased toward interpolation on a super-resolution task, and using it alone to rank
//  upscale tiers is a category error, not a close call.
//
//  ⚠️ **The no-reference arm is not the "right" metric replacing a "wrong" one, and must never be
//  presented that way.** NR-IQA is blind to content identity — swap in a *different* beautiful image
//  and the score goes up — and it is trivially gamed by over-sharpening and halos (GAP-PROGRAM §V, and
//  the standing "never put NR-delta in a training loop" rule). It answers *"does this look like a good
//  image?"*; SSIMULACRA2 answers *"is this the same image?"*. **Report both, side by side, and let the
//  disagreement stand: on this task the disagreement IS the finding.**
//
//  ⚠️ SigLIP2 is PNG-only at the boundary. `SigLIP2IQAPackage.decodeToPixelBuffer` runs
//  `CGImageSourceCreateWithData`, so a `.rawBGRA8` artifact fails as a decode error deep inside the
//  package rather than here. Encode PNG, per `EngineImageCoding`'s stated caller-decides contract.
//

import CoreGraphics
import Foundation
import MLXServeCore
import MLXToolKit

extension ForgeEngineService {

    /// Score one image 0–1 with SigLIP2 NR-IQA "restoration-pays" semantics — **low where enhancement
    /// would help, high on clean content** — using no reference at all.
    ///
    /// - Throws: `DiagnosticsError.notRegistered` when the package did not register, carrying the
    ///   registration failure text: a license refusal and a missing backbone snapshot are different
    ///   problems, and a nil would make them the same problem.
    public func scoreNoReferenceQuality(_ image: CGImage) async throws -> Double {
        guard let packageID = packageID(for: .qualityScore) else {
            throw DiagnosticsError.notRegistered(.qualityScore, outcome(for: .qualityScore)?.failure)
        }
        let encoded = try EngineImageCoding.pngImage(from: image)
        _ = try await engine.prepare(.imageQualityScore, package: packageID)
        let response = try await engine.run(ImageQualityScoreRequest(image: encoded), package: packageID)
        guard let score = response as? ImageQualityScoreResponse else {
            throw DiagnosticsError.unexpectedResponse(.qualityScore)
        }
        return Double(score.score)
    }
}
