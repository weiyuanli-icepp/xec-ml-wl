/*
PrepareRealData.C
- Reads Rec Trees (Real Data)
- Applies Event Selection (PhysicsSelection + EvstatGamma==0)
- Calculates ML Features (Relative Npho, Time)
- Calculates Geometry (xyzVTX from Reco, emiVec, emiAng)
- outputs a flat tree compatible with the Python Inference script.

Usage:
$ ./meganalyzer -b -q -I /path/to/PrepareRealData.C+
*/

#include <TROOT.h>
#include <TMath.h>
#include <TChain.h>
#include <TFile.h>
#include <TVector3.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TSystem.h>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

#if !defined(__CLING__) || defined(__ROOTCLING__)
#   include <ROMETreeInfo.h>
#   include "include/generated/MEGAllFolders.h"
#   include "include/generated/MEGMEGParameters.h"
#   include "include/generated/MEGRecData.h"
#   include "MEGGLBParticlesDataCombine.h"
#   include "include/generated/MEGEventHeader.h"
#   include "include/generated/MEGXECTimeFitResult.h"
#   include "include/generated/MEGXECPosLocalFitResult.h"
#   include "include/generated/MEGXECEneTotalSumRecResult.h"
#   include "include/generated/MEGXECPMCluster.h"
#   include "xec/xectools.h"
#   include "glb/MEGPhysicsSelection.h"
#   include "units/MEGSystemOfUnits.h"
#else
   class ROMETreeInfo;
   class MEGEventHeader;
   class MEGRecData;
   class MEGGLBParticlesDataCombine;
   class MEGXECPosLocalFitResult;
   class MEGXECTimeFitResult;
   class MEGXECEneTotalSumRecResult;
   class MEGXECPMCluster;
#endif

using namespace MEG;

// Helper function declarations
std::map<Int_t, TString> PrepareRunList(Int_t startRun, Int_t maxNRuns, TString suffix, TString dir);

void PrepareRealData(Int_t sRun = 430000, Int_t nfile = 2000, TString fileSuffix = "_open.root")
{
   // =========================================================================
   // 1. CONFIGURATION
   // =========================================================================
   // Input Directory
   TString inputrecdir = "/data/project/meg/offline/run/"; 
   TString runList = ""; 

   // Geometry Init
   XECTOOLS::InitXECGeometryParameters(XECTOOLS::UVWDefinition::Global,
                                       64.84, 106.27, 67.03, 96., 125.52, 
                                       0, 0, 0);

   // =========================================================================
   // 2. FILE LOADING
   // =========================================================================
   std::map<Int_t, TString> files = PrepareRunList(inputrecdir, fileSuffix, sRun, nfile);
   
   if (files.empty()) {
       std::cout << "[Error] No files found matching suffix '" << fileSuffix << "' starting from run " << sRun << std::endl;
       return;
   }

   Int_t startRun = files.begin()->first;
   Int_t lastRun = files.rbegin()->first;

   TChain *rec = new TChain("rec");
   for (auto const& [run, file] : files) {
      rec->Add(file);
      std::cout << "Added: " << file << std::endl;
   }

   // =========================================================================
   // 3. OUTPUT SETUP
   // =========================================================================
   TFile outputFile(Form("DataGammaAngle_%d-%d.root", startRun, lastRun), "RECREATE", "Real Data ML Input");
   TTree *tree = new TTree("tree", "Real Data Inference Tree");

   static const int kXECNChan = 4760;

   Int_t   run_out = 0;
   Int_t   event_out = 0;

   Float_t xyzRecoFI[3];  // from XEC PosLocalFit
   Float_t xyzRecoLC[3];  // from XEC PosLocalFit Light Center
   Float_t xyzVTX[3];     // from Reco Positron XYZ
   Float_t emiVec[3];     // Reco Gamma XYZ - xyzVTX
   Float_t emiAng[2];     // Calculated from emiVec
   Float_t uvwRecoFI[3];
   Float_t uvwRecoLC[3];
   Float_t energyReco  = 1e10f;
   Float_t timeReco    = 1e10f;
   Float_t npho[kXECNChan];
   Float_t nphe[kXECNChan];
   Float_t time[kXECNChan];
   Float_t relative_npho[kXECNChan];
   Float_t relative_time[kXECNChan];

   Short_t ch_npho_max = -1, ch_time_min = -1;
   Float_t npho_max_used = 1e10f, time_min_used = 1e10f;

   // Branch Definitions
   tree->Branch("run",   &run_out,   "run/I");
   tree->Branch("event", &event_out, "event/I");
   tree->Branch("xyzRecoFI", xyzRecoFI, "xyzRecoFI[3]/F");
   tree->Branch("xyzRecoLC", xyzRecoLC, "xyzRecoLC[3]/F");
   tree->Branch("xyzVTX",    xyzVTX,    "xyzVTX[3]/F");
   tree->Branch("emiVec",    emiVec,    "emiVec[3]/F");
   tree->Branch("emiAng",    emiAng,    "emiAng[2]/F");
   tree->Branch("uvwRecoFI", uvwRecoFI, "uvwRecoFI[3]/F");
   tree->Branch("uvwRecoLC", uvwRecoLC, "uvwRecoLC[3]/F");
   tree->Branch("energyReco",  &energyReco,  "energyReco/F");
   tree->Branch("timeReco",    &timeReco,    "timeReco/F");
   tree->Branch("npho", npho, Form("npho[%d]/F", kXECNChan));
   tree->Branch("time", time, Form("time[%d]/F", kXECNChan));
   tree->Branch("relative_npho", relative_npho, Form("relative_npho[%d]/F", kXECNChan));
   tree->Branch("relative_time", relative_time, Form("relative_time[%d]/F", kXECNChan));
   tree->Branch("ch_npho_max", &ch_npho_max, "ch_npho_max/S");
   tree->Branch("ch_time_min", &ch_time_min, "ch_time_min/S");
   tree->Branch("npho_max_used", &npho_max_used, "npho_max_used/F");
   tree->Branch("time_min_used", &time_min_used, "time_min_used/F");

   // =========================================================================
   // 4. READERS & SELECTION SETUP
   // =========================================================================
   TTreeReader reader(rec);
   
   // Basic Info
   TTreeReaderValue<ROMETreeInfo> infoRV(reader, "Info.");
   TTreeReaderValue<MEGEventHeader> headerRV(reader, "eventheader.");
   
   // Reconstruction Data
   TTreeReaderValue<MEGRecData>                 recoRV(reader, "reco."); 
   TTreeReaderArray<MEGXECPosLocalFitResult>    xecposfitRA(reader, "xecposlfit");
   TTreeReaderArray<MEGXECTimeFitResult>        xectimefitRA(reader, "xectimefit");
   TTreeReaderArray<MEGXECEneTotalSumRecResult> xecenerecRA(reader, "xecenetotalsum");
   TTreeReaderArray<MEGXECPMCluster>            xecclRA(reader, "xeccl");
   TTreeReaderValue<MEGGLBParticlesDataCombine> combinationRV(reader, "combination.");

   // Selection
   MEGPhysicsSelection selector(kFALSE, 0, kTRUE);
   selector.SetThresholds(EBeamPeriodID::kBeamPeriod2022, kTRUE);
   selector.fTimePositronGamma[0] = -5*nanosecond; 
   selector.fTimePositronGamma[1] = 5*nanosecond; 
   selector.fCosThetaPositronGamma[0] = -1;
   selector.fCosThetaPositronGamma[1] = 1;
   selector.fThetaPositronGamma[0] = -2*radian;
   selector.fThetaPositronGamma[1] = 2*radian;
   selector.fPhiPositronGamma[0] = -2*radian;
   selector.fPhiPositronGamma[1] = 2*radian;
   selector.fEPositron[0] = 40*MeV;
   selector.fEPositron[1] = 60*MeV;
   selector.fEGamma[0] = 35*MeV;
   selector.fEGamma[1] = 65*MeV; 

   // =========================================================================
   // 5. EVENT LOOP
   // =========================================================================
   Int_t nTotal = rec->GetEntries();
   Int_t nProcessed = 0;
   Int_t nSelected = 0;

   std::cout << "Starting loop over " << nTotal << " events..." << std::endl;

   while (reader.Next()) {
      if (nProcessed % 1000 == 0) std::cout << "Processing " << nProcessed << " / " << nTotal << std::endl;
      nProcessed++;

      // --- 1. Trigger Selection ---
      if (headerRV->Getmask() != 0) continue;

      // --- 2. Physics Selection ---
      std::vector<Bool_t> selected;
      MEGGLBParticlesDataCombine* pComb = combinationRV.Get();
      MEGRecData* pReco = recoRV.Get();

      selector.CombinedSelection(selected, pComb, pReco);

      Int_t nPair = std::count(selected.begin(), selected.end(), kTRUE);
      if (!nPair) continue;

      Int_t bestPairIndex = -1;
      for (size_t i=0; i<selected.size(); i++) {
         if (selected[i]) {
            bestPairIndex = i;
            break;
         }
      }

      Int_t gammaIdx    = pComb->GetGammaIndexAt(bestPairIndex);
      Int_t positronIdx = pComb->GetPositronIndexAt(bestPairIndex);

      // --- 3. Pilup Selection ---
      if (pReco->GetEvstatGammaAt(gammaIdx) != 0) continue;

      // --- 4. Data Extraction ---
      run_out   = infoRV->GetRunNumber();
      event_out = infoRV->GetEventNumber();

      if (xecposfitRA.GetSize() < 1 || xectimefitRA.GetSize() < 1 || xecenerecRA.GetSize() < 1) continue;

      // --- VTX and Emission Vector Calculation ---
      // VTX from Positron Reco
      xyzVTX[0] = pReco->GetXPositronAt(positronIdx);
      xyzVTX[1] = pReco->GetYPositronAt(positronIdx);
      xyzVTX[2] = pReco->GetZPositronAt(positronIdx);

      // Emission Vector: Gamma Reco Position - VTX
      Double_t xG = pReco->GetXGammaAt(gammaIdx);
      Double_t yG = pReco->GetYGammaAt(gammaIdx);
      Double_t zG = pReco->GetZGammaAt(gammaIdx);
      TVector3 vEmi(xG - xyzVTX[0], yG - xyzVTX[1], zG - xyzVTX[2]);
      if (vEmi.Mag() > 0) vEmi = vEmi.Unit();

      emiVec[0] = vEmi.X();
      emiVec[1] = vEmi.Y();
      emiVec[2] = vEmi.Z();

      // Photon Angles
      emiAng[0] = vEmi.Theta() * 180.0 / TMath::Pi();
      emiAng[1] = TMath::ATan2(vEmi.Y(), -vEmi.X()) * 180.0 / TMath::Pi();

      // XEC Reco Variables
      const auto& aPos = xecposfitRA.At(gammaIdx);
      const auto& aEne = xecenerecRA.At(gammaIdx);
      const auto& aTime = xectimefitRA.At(gammaIdx);

      energyReco = aEne.Getenergy();

      for(int i=0; i<3; i++) {
         xyzRecoFI[i] = aPos.GetxyzAt(i);
         uvwRecoFI[i] = aPos.GetuvwAt(i);
      }

      // Light Center Conversion
      Double_t uvwLC_d[3], xyzLC_d[3];
      for(int i=0; i<3; i++) {
         uvwLC_d[i] = aPos.GetuvwLightCenterAt(i);
         uvwRecoLC[i] = static_cast<Float_t>(uvwLC_d[i]);
      }
      XECTOOLS::UVW2XYZ(uvwLC_d, xyzLC_d);
      for(int i=0; i<3; i++) xyzRecoLC[i] = static_cast<Float_t>(xyzLC_d[i]);

      // Time Reco
      TVector3 xyzRecV(aPos.GetxyzAt(0), aPos.GetxyzAt(1), aPos.GetxyzAt(2));
      TVector3 xyzEmV(xyzVTX);
      TVector3 flight = xyzRecV - xyzEmV;
      timeReco = 1e9 * aTime.Gettime() - flight.Mag() / 29.9792458;

      // --- Npho / Time Array Extraction ---
      for(int ch=0; ch<kXECNChan; ch++) {
         npho[ch] = 1e10f;
         nphe[ch] = 1e10f;
         time[ch] = 1e10f;
         relative_npho[ch] = 1e10f;
         relative_time[ch] = 1e10f;
      }

      if (xecclRA.GetSize() == kXECNChan) {
         for (int ch=0; ch<kXECNChan; ++ch) {
            MEGXECPMCluster &cl = xecclRA.At(ch);
            npho[ch] = cl.GetnphoAt(0);
            nphe[ch] = cl.GetnpheAt(0);
            time[ch] = cl.GettpmAt(0);
         }
      }

      // --- Calculate Relative Arrays ---
      Float_t val_max  = -std::numeric_limits<Float_t>::infinity();
      Float_t min_time =  std::numeric_limits<Float_t>::infinity();
       
      ch_npho_max = -1;
      ch_time_min = -1;
      npho_max_used = 1e10f;
      time_min_used = 1e10f;

      // 1. Find Max/Min
      for (Int_t ch=0; ch<kXECNChan; ++ch) {
         const Float_t v = npho[ch];
         const Float_t n = nphe[ch];
         if (std::isfinite(v) && v>=0.0f && v < 1e9f) {
            if (v > val_max) {
               val_max = v;
               ch_npho_max = ch;
            }
         }
         const Float_t t = time[ch];
         if (std::isfinite(t) && t < min_time && n > 50) { 
            min_time = t;
            ch_time_min = ch;
         }
      }

      npho_max_used = (std::isfinite(val_max) && val_max < 1e9f) ? val_max : 1e10f;
      time_min_used = (std::isfinite(min_time) && min_time < 1e9f) ? min_time : 1e10f;

      if (npho_max_used == 1e10f || time_min_used == 1e10f) continue;

      // 2. Fill Relatives
      for (Int_t ch=0; ch<kXECNChan; ++ch) {
         if (ch_npho_max >= 0 && std::isfinite(val_max) && std::fabs(val_max) > 0) {
            relative_npho[ch] = (std::isfinite(npho[ch]) && npho[ch] < 1e9f) ? (npho[ch] / val_max) : 1e10f;
         }
           
         if (std::isfinite(min_time) && min_time < 1e9f && std::isfinite(time[ch]) && time[ch] < 1e9f) {
            relative_time[ch] = time[ch] - min_time;
         }
      }

      // Fill Tree
      tree->Fill();
      nSelected++;
   }

   tree->Write();
   outputFile.Close();
   
   std::cout << "Done! Selected " << nSelected << " events." << std::endl;
   std::cout << "Output: " << outputFile.GetName() << std::endl;
}

//______________________________________________________________________________
// Helper: Prepare File List
std::map<Int_t, TString> PrepareRunList(TString dir, TString suffix, Int_t startRun, Int_t maxNRuns)
{
   std::map<Int_t, TString> files;
   int found = 0;
   int run = startRun;
   int consecutive_misses = 0;

   while (found < (maxNRuns > 0 ? maxNRuns : 10000) && consecutive_misses < 1000) {
      TString path = Form("%s/%03dxxx/rec%06d%s", dir.Data(), run/1000, run, suffix.Data());
      if (gSystem->AccessPathName(path)) {
         path = Form("%s/rec%06d%s", dir.Data(), run, suffix.Data());
      }
      if (!gSystem->AccessPathName(path)) {
         files[run] = path;
         found++;
         consecutive_misses = 0;
      } else {
         consecutive_misses++;
      }
      run++;
   }
   return files;
}