// CEXPreprocess.C — Real-data CEX preprocessing for energy regressor validation
//
// Standalone macro (no MEG framework or MySQL needed). Reads rec files using
// leaf-level access via GetLeaf()->GetValue().
//
// Usage:
//   root -l -b -q 'others/CEXPreprocess.C(557545, 1, 13)'
//   root -l -b -q 'others/CEXPreprocess.C(557545, 100, 13, ".", "dead_channels.txt")'
//
// Arguments:
//   sRun            — starting run number
//   nfiles          — number of consecutive runs to try
//   patchnumber     — CEX patch number (for output filename only)
//   outputDir       — output directory (default: current directory)
//   deadChannelFile — text file with one dead channel index per line
//                     (generate with: python -m lib.db_utils RUN -o dead.txt)
//                     If empty, dead channels are detected from sentinel values.
//
// Note: Without DB access, all runs in [sRun, sRun+nfiles) are attempted.
//       Non-existent rec files are skipped. The energy window cut (54-57 MeV)
//       naturally rejects non-CEX events.

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <set>

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TMath.h"
#include "Riostream.h"

void CEXPreprocess(Int_t sRun, Int_t nfiles, Int_t patchnumber,
                   TString outputDir = ".", TString deadChannelFile = "")
{
  static const Int_t kXECNChan = 4760;

  // --- π⁰ kinematics constants (GeV) ---
  const Float_t Epi0 = 0.1378;    // π⁰ kinetic energy + mass at rest in MEG target
  const Float_t mpi0 = 0.13497;   // π⁰ mass (GeV/c²)

  // --- CEX23 energy window for 55 MeV peak (GeV) ---
  const Float_t min55 = 0.054;
  const Float_t max55 = 0.057;

  // --- Load dead channel list from file (if provided) ---
  std::set<Int_t> deadFromFile;
  if (deadChannelFile.Length() > 0) {
    std::ifstream fin(deadChannelFile.Data());
    if (!fin.is_open()) {
      std::cerr << "Warning: cannot open dead channel file: "
                << deadChannelFile << std::endl;
    } else {
      std::string line;
      while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        Int_t idx = std::stoi(line);
        if (idx >= 0 && idx < kXECNChan) deadFromFile.insert(idx);
      }
      fin.close();
      std::cout << "Loaded " << deadFromFile.size()
                << " dead channels from " << deadChannelFile << std::endl;
    }
  }

  // --- Output file ---
  TString foutname = Form("%s/CEX23_patch%d_r%d_n%d.root",
                          outputDir.Data(), patchnumber, sRun, nfiles);
  TFile *fout = new TFile(foutname.Data(), "RECREATE",
                          "CEX real data for regressor");

  // --- Output variables ---
  Int_t   run = 0, event = 0;
  Float_t energyTruth = 1e10f;
  Float_t energyReco  = 1e10f;
  Float_t timeTruth   = 1e10f;

  Float_t uvwRecoFI[3];
  Float_t uvwTruth[3];
  Float_t xyzTruth[3];
  Float_t emiAng[2];
  Float_t emiVec[3];
  Float_t xyzVTX[3];

  Float_t npho[kXECNChan];
  Float_t nphe[kXECNChan];
  Float_t time_arr[kXECNChan];
  Float_t relative_time[kXECNChan];

  Short_t ch_npho_max = -1, ch_time_min = -1;
  Float_t npho_max_used = 1e10f, time_min_used = 1e10f;

  // Dead channel tracking
  Bool_t  dead[kXECNChan];
  Int_t   nDeadChannels = 0;

  // CEX-specific metadata
  Float_t Ebgo = 0, Angle = 0;
  Int_t   gstatus = 0;

  // --- Create output tree (same name as MCXECPreprocess) ---
  TTree *tree = new TTree("tree",
      "CEX real data for Gamma variables prediction ML validation");

  // Branches matching MCXECPreprocess format (read by lib/dataset.py)
  tree->Branch("run",   &run,   "run/I");
  tree->Branch("event", &event, "event/I");

  tree->Branch("energyReco",  &energyReco,  "energyReco/F");
  tree->Branch("energyTruth", &energyTruth, "energyTruth/F");
  tree->Branch("timeTruth",   &timeTruth,   "timeTruth/F");

  tree->Branch("uvwRecoFI", uvwRecoFI, "uvwRecoFI[3]/F");
  tree->Branch("uvwTruth",  uvwTruth,  "uvwTruth[3]/F");
  tree->Branch("xyzTruth",  xyzTruth,  "xyzTruth[3]/F");
  tree->Branch("emiAng",    emiAng,    "emiAng[2]/F");
  tree->Branch("emiVec",    emiVec,    "emiVec[3]/F");
  tree->Branch("xyzVTX",    xyzVTX,    "xyzVTX[3]/F");

  tree->Branch("npho", npho, Form("npho[%d]/F", kXECNChan));
  tree->Branch("nphe", nphe, Form("nphe[%d]/F", kXECNChan));
  tree->Branch("time", time_arr, Form("time[%d]/F", kXECNChan));
  tree->Branch("relative_time", relative_time,
               Form("relative_time[%d]/F", kXECNChan));

  tree->Branch("ch_npho_max",  &ch_npho_max,  "ch_npho_max/S");
  tree->Branch("ch_time_min",  &ch_time_min,  "ch_time_min/S");
  tree->Branch("npho_max_used", &npho_max_used, "npho_max_used/F");
  tree->Branch("time_min_used", &time_min_used, "time_min_used/F");

  // Dead channel branches
  tree->Branch("dead",  dead,           Form("dead[%d]/O", kXECNChan));
  tree->Branch("nDead", &nDeadChannels, "nDead/I");

  // CEX-specific branches (useful for diagnostics/cuts)
  tree->Branch("Ebgo",    &Ebgo,    "Ebgo/F");
  tree->Branch("Angle",   &Angle,   "Angle/F");
  tree->Branch("gstatus", &gstatus, "gstatus/I");

  // --- Run loop ---
  TString inputrecdir = "/data/project/meg/offline/run";
  Int_t totalEvents = 0;
  Int_t runsProcessed = 0;

  for (Int_t iRun = sRun; iRun < sRun + nfiles; iRun++) {

    // --- Open rec file ---
    TString filename = Form("%s/%3dxxx/rec%05d.root",
                            inputrecdir.Data(), iRun / 1000, iRun);
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
      if (nfiles == 1)
        std::cerr << "Warning: could not open " << filename << std::endl;
      delete file;
      continue;
    }

    TTree *rec = file->Get<TTree>("rec");
    if (!rec) {
      std::cerr << "Warning: 'rec' tree not found in " << filename << std::endl;
      file->Close();
      delete file;
      continue;
    }

    runsProcessed++;
    std::cout << "Run " << iRun << " (" << filename << ")" << std::endl;

    // --- Set dead channel mask ---
    for (Int_t ch = 0; ch < kXECNChan; ch++) dead[ch] = kFALSE;
    nDeadChannels = 0;

    if (!deadFromFile.empty()) {
      // From file
      for (Int_t idx : deadFromFile) {
        dead[idx] = kTRUE;
        nDeadChannels++;
      }
    }
    // If no file provided, dead channels will be detectable downstream
    // from sentinel values (npho = 1e10) in the output.
    std::cout << "  Dead channels: " << nDeadChannels
              << (deadFromFile.empty() ? " (detect from sentinels downstream)" : " (from file)")
              << std::endl;

    // --- Resolve leaf pointers (standalone: no MEG headers needed) ---
    // ROME convention: GetXXX() → member fXXX stored as leaf "branch.fXXX"
    rec->GetEntry(0);

    TLeaf *leaf_mask       = rec->GetLeaf("eventheader.fmask");
    TLeaf *leaf_EGamma     = rec->GetLeaf("reco.fEGamma");
    TLeaf *leaf_EvstatGamma = rec->GetLeaf("reco.fEvstatGamma");
    TLeaf *leaf_UGamma     = rec->GetLeaf("reco.fUGamma");
    TLeaf *leaf_VGamma     = rec->GetLeaf("reco.fVGamma");
    TLeaf *leaf_WGamma     = rec->GetLeaf("reco.fWGamma");
    TLeaf *leaf_npho       = rec->GetLeaf("xeccl.fnpho");
    TLeaf *leaf_nphe       = rec->GetLeaf("xeccl.fnphe");
    TLeaf *leaf_tpm        = rec->GetLeaf("xeccl.ftpm");
    TLeaf *leaf_openAngle  = rec->GetLeaf("bgocexresult.fopeningAngle");
    TLeaf *leaf_bgoEnergy  = rec->GetLeaf("bgocexresult.fbgoEnergy");

    // Validate critical leaves
    if (!leaf_mask || !leaf_EGamma || !leaf_npho || !leaf_tpm || !leaf_openAngle) {
      std::cerr << "  ERROR: Missing critical leaves in rec tree:" << std::endl;
      if (!leaf_mask)      std::cerr << "    - eventheader.fmask NOT FOUND" << std::endl;
      if (!leaf_EGamma)    std::cerr << "    - reco.fEGamma NOT FOUND" << std::endl;
      if (!leaf_npho)      std::cerr << "    - xeccl.fnpho NOT FOUND" << std::endl;
      if (!leaf_tpm)       std::cerr << "    - xeccl.ftpm NOT FOUND" << std::endl;
      if (!leaf_openAngle) std::cerr << "    - bgocexresult.fopeningAngle NOT FOUND" << std::endl;
      std::cerr << "  Use rec->Print() to inspect leaf names. Skipping run." << std::endl;
      file->Close();
      delete file;
      continue;
    }

    // Determine stride for xeccl array members (fnpho is [nChannels][nFitResults])
    Int_t nphoLen = leaf_npho->GetLen();
    Int_t stride = (nphoLen >= kXECNChan) ? nphoLen / kXECNChan : 1;
    if (stride > 1)
      std::cout << "  xeccl stride = " << stride << " (nFitResults per channel)" << std::endl;

    // --- Event loop ---
    Int_t Nevent = rec->GetEntries();
    Int_t eventsThisRun = 0;
    for (Int_t ev = 0; ev < Nevent; ev++) {
      // Progress bar
      if (ev == 0)
        std::cout << "  [Processing " << Nevent << " events] [" << std::flush;
      if (Nevent > 20 && ev % (Nevent / 20) == 0)
        std::cout << "=" << std::flush;
      if (ev == Nevent - 1)
        std::cout << "]" << std::endl;

      rec->GetEntry(ev);

      // Read trigger mask
      Int_t triggermask = (Int_t)leaf_mask->GetValue(0);

      // Keep only physics triggers (50, 51); skip pedestal (63) and others
      if (triggermask != 50 && triggermask != 51) continue;

      // --- Reconstruction quantities ---
      Float_t erec = (Float_t)leaf_EGamma->GetValue(0);

      // Energy window: 55 MeV peak only
      if (erec < min55 || erec > max55) continue;

      // Opening angle and Etrue from 2-body kinematics
      Float_t openingangle = (Float_t)leaf_openAngle->GetValue(0);
      Float_t sqrtarg = 0.25f * Epi0 * Epi0
                        - mpi0 * mpi0 / (2.0f * (1.0f - TMath::Cos(openingangle * TMath::Pi() / 180.0f)));
      if (sqrtarg < 0) continue;  // unphysical — skip
      Float_t etrue = Epi0 / 2.0f - TMath::Sqrt(sqrtarg);

      // --- Fill output variables ---
      run   = iRun;
      event = ev;

      energyTruth = etrue;
      energyReco  = erec;
      gstatus     = leaf_EvstatGamma ? (Int_t)leaf_EvstatGamma->GetValue(0) : 0;

      uvwRecoFI[0] = leaf_UGamma ? (Float_t)leaf_UGamma->GetValue(0) : 1e10f;
      uvwRecoFI[1] = leaf_VGamma ? (Float_t)leaf_VGamma->GetValue(0) : 1e10f;
      uvwRecoFI[2] = leaf_WGamma ? (Float_t)leaf_WGamma->GetValue(0) : 1e10f;

      Angle = openingangle;
      Ebgo  = leaf_bgoEnergy ? (Float_t)leaf_bgoEnergy->GetValue(0) : 1e10f;

      // Truth branches not available for real data — fill with 1e10 sentinel
      timeTruth = 1e10f;
      for (int i = 0; i < 3; i++) {
        uvwTruth[i] = 1e10f;
        xyzTruth[i] = 1e10f;
        xyzVTX[i]   = 1e10f;
        emiVec[i]    = 1e10f;
      }
      emiAng[0] = 1e10f;
      emiAng[1] = 1e10f;

      // --- Read per-channel npho, nphe, time ---
      for (Int_t ch = 0; ch < kXECNChan; ch++) {
        npho[ch] = 1e10f;
        nphe[ch] = 1e10f;
        time_arr[ch] = 1e10f;
        relative_time[ch] = 1e10f;
      }

      if (leaf_npho->GetLen() >= kXECNChan) {
        for (Int_t ch = 0; ch < kXECNChan; ch++) {
          npho[ch]     = (Float_t)leaf_npho->GetValue(ch * stride);
          if (leaf_nphe) nphe[ch] = (Float_t)leaf_nphe->GetValue(ch * stride);
          time_arr[ch] = (Float_t)leaf_tpm->GetValue(ch * stride);
        }
      }

      // --- Compute relative_time and find max npho ---
      Float_t val_max  = -std::numeric_limits<Float_t>::infinity();
      Float_t min_time =  std::numeric_limits<Float_t>::infinity();
      ch_npho_max = -1;
      ch_time_min = -1;

      for (Int_t ch = 0; ch < kXECNChan; ch++) {
        Float_t v = npho[ch];
        Float_t n = nphe[ch];
        if (std::isfinite(v) && v >= 0.0f && v < 1e9f) {
          if (v > val_max) {
            val_max = v;
            ch_npho_max = ch;
          }
        }
        Float_t t = time_arr[ch];
        if (std::isfinite(t) && t < min_time && n > 50) {
          min_time = t;
          ch_time_min = ch;
        }
      }

      npho_max_used = (std::isfinite(val_max) && val_max < 1e9f) ? val_max : 1e10f;
      time_min_used = (std::isfinite(min_time) && min_time < 1e9f) ? min_time : 1e10f;

      // Skip events with no valid npho or time
      if (npho_max_used > 1e9f || time_min_used > 1e9f) continue;

      // Fill relative_time
      for (Int_t ch = 0; ch < kXECNChan; ch++) {
        if (std::isfinite(min_time) && min_time < 1e9f &&
            std::isfinite(time_arr[ch]) && time_arr[ch] < 1e9f) {
          relative_time[ch] = time_arr[ch] - min_time;
        } else {
          relative_time[ch] = 1e10f;
        }
      }

      tree->Fill();
      eventsThisRun++;
    } // end event loop

    totalEvents += eventsThisRun;
    std::cout << "  -> " << eventsThisRun << " events selected" << std::endl;

    file->Close();
    delete file;

  } // end run loop

  // --- Write output ---
  fout->cd();
  tree->Write();
  fout->Close();

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Runs processed: " << runsProcessed << std::endl;
  std::cout << "Events selected: " << totalEvents << std::endl;
  std::cout << "Output: " << foutname << std::endl;
}
