//
//  FillRouteExports.swift
//  ForgeCore
//
//  The fill route's pixel steps live in EraseKit's `EraseFill` product since ForgeCore 0.47.0 (AB-A-0180): ONE
//  implementation, shared with pixel-agency-normalizer (Gaea, Pixel Agency Desktop). It moved verbatim. ForgeCore
//  re-exports exactly the four route types, so `import ForgeCore` keeps compiling `FillRoute.composite(…)`,
//  `FillPixels(…)` and the rest (Pixel Agency Editor's PixelAgencyForge) with no source change and no new import.
//  `GenerativeFillLiveTests/bakeOffReproduces` is the pixel-for-pixel guard on the route.
//

@_exported import struct EraseFill.FillPixels
@_exported import struct EraseFill.FillMask
@_exported import struct EraseFill.FillBox
@_exported import enum EraseFill.FillRoute
