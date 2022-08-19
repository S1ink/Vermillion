-- premake5.lua
workspace "AnalysisApp"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "AnalysisApp"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "AnalysisApp"