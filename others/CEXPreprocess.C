// CEXPreprocess.C — Real-data CEX preprocessing for energy regressor validation
//
// Reads reconstructed π⁰ CEX calibration runs and outputs a ROOT tree
// in the same format as MCXECPreprocess.C so the regressor data loader
// (lib/dataset.py) can consume it directly.
//
// Usage:
//   root -l -b -q 'CEXPreprocess.C(557545, 100, 13)'
//   root -l -b -q 'CEXPreprocess.C(557545, 100, 13, "/my/output/dir")'
//
// Arguments:
//   sRun        — starting run number
//   nfiles      — number of consecutive runs to scan
//   patchnumber — CEX patch number to select
//   outputDir   — output directory (default: current directory)

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TMath.h"
#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TClonesArray.h"
#include "Riostream.h"

#include <ROMETreeInfo.h>
#include "/data/user/ext-li_w1/meghome/offline/analyzer/include/generated/MEGAllFolders.h"

void CEXPreprocess(Int_t sRun, Int_t nfiles, Int_t patchnumber,
                   TString outputDir = ".")
{
  static const Int_t kXECNChan = 4760;

  // --- π⁰ kinematics constants (GeV) ---
  const Float_t Epi0 = 0.1378;    // π⁰ kinetic energy + mass at rest in MEG target
  const Float_t mpi0 = 0.13497;   // π⁰ mass (GeV/c²)

  // --- CEX23 energy window for 55 MeV peak (GeV) ---
  const Float_t min55 = 0.054;
  const Float_t max55 = 0.057;

  // --- Connect to MySQL for geometry and run catalog ---
  TSQLServer *SQLServer = TSQLServer::Connect("mysql://meg.sql.psi.ch",
                                               "meg_ro", "readonly");
  if (!SQLServer || SQLServer->IsZombie()) {
    std::cerr << "Error connecting to MySQL server" << std::endl;
    return;
  }
  SQLServer->SelectDataBase("MEG2");

  // --- Load sensor geometry from DB (for index validation only) ---
  // We do not need positions for the output (no solid angle computation).
  std::cout << "Loading XEC geometry from database..." << std::endl;
  TSQLResult *geoRes = SQLServer->Query(
      "SELECT idx FROM XECGeometry WHERE id = 3;");
  Int_t nGeomLoaded = 0;
  if (geoRes) {
    while (TSQLRow *row = geoRes->Next()) {
      Int_t ch = TString(row->GetField(0)).Atoi();
      if (ch >= 0 && ch < kXECNChan) nGeomLoaded++;
      delete row;
    }
    delete geoRes;
  }
  std::cout << "  Loaded " << nGeomLoaded << " channels from XECGeometry" << std::endl;

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

    // --- Query RunCatalog ---
    TString query;
    query.Form("SELECT Junk, RunDescription, Physics FROM RunCatalog WHERE id=%d;",
               iRun);
    TSQLResult *SQLResult = SQLServer->Query(query.Data());
    if (!SQLResult) continue;
    TSQLRow *SQLRow = SQLResult->Next();
    if (!SQLRow) { delete SQLResult; continue; }

    Int_t junk = SQLRow->GetFieldLength(0) ? atoi(SQLRow->GetField(0)) : 1;
    TString rundescription = SQLRow->GetField(1);
    delete SQLRow;
    delete SQLResult;

    // Skip junk runs
    if (junk) continue;

    // Must be "Pi0 CEX" run
    if (!rundescription.Contains("Pi0 CEX", TString::kIgnoreCase)) continue;

    // Must match requested patch number
    Ssiz_t patchPos = rundescription.Index("patch number", 11, TString::kIgnoreCase);
    if (patchPos == kNPOS) continue;
    TString patchStr = rundescription(patchPos + 12, 3);
    Int_t patchNumber = -1;
    sscanf(patchStr.Data(), "%d", &patchNumber);
    if (patchNumber != patchnumber) continue;

    // --- Open rec file ---
    TString filename = Form("%s/%3dxxx/rec%05d.root",
                            inputrecdir.Data(), iRun / 1000, iRun);
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
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
    std::cout << "Run " << iRun << " | Patch " << patchNumber
              << " | " << rundescription << std::endl;

    // --- Query dead channels from DB for this run ---
    // Chain: RunCatalog → XECConf_id → XECPMStatusDB_id → XECPMStatus_id → dead channels
    for (Int_t ch = 0; ch < kXECNChan; ch++) dead[ch] = kFALSE;
    nDeadChannels = 0;

    {
      // Step 1: RunCatalog → XECConf_id
      TString qConf;
      qConf.Form("SELECT XECConf_id FROM RunCatalog WHERE id=%d;", iRun);
      TSQLResult *rConf = SQLServer->Query(qConf.Data());
      Int_t xecConfId = -1;
      if (rConf) {
        TSQLRow *rowConf = rConf->Next();
        if (rowConf && rowConf->GetField(0))
          xecConfId = TString(rowConf->GetField(0)).Atoi();
        delete rowConf;
        delete rConf;
      }

      if (xecConfId >= 0) {
        // Step 2: XECConf → XECPMStatusDB_id
        TString qStatusDB;
        qStatusDB.Form("SELECT XECPMStatusDB_id FROM XECConf WHERE id=%d;", xecConfId);
        TSQLResult *rStatusDB = SQLServer->Query(qStatusDB.Data());
        Int_t pmStatusDBId = -1;
        if (rStatusDB) {
          TSQLRow *rowSDB = rStatusDB->Next();
          if (rowSDB && rowSDB->GetField(0))
            pmStatusDBId = TString(rowSDB->GetField(0)).Atoi();
          delete rowSDB;
          delete rStatusDB;
        }

        if (pmStatusDBId >= 0) {
          // Step 3: XECPMStatusDB → XECPMStatus_id
          TString qStatusId;
          qStatusId.Form("SELECT XECPMStatus_id FROM XECPMStatusDB WHERE id=%d;", pmStatusDBId);
          TSQLResult *rStatusId = SQLServer->Query(qStatusId.Data());
          Int_t pmStatusId = -1;
          if (rStatusId) {
            TSQLRow *rowSId = rStatusId->Next();
            if (rowSId && rowSId->GetField(0))
              pmStatusId = TString(rowSId->GetField(0)).Atoi();
            delete rowSId;
            delete rStatusId;
          }

          if (pmStatusId >= 0) {
            // Step 4: XECPMStatus WHERE IsBad=1 → dead channel indices
            TString qDead;
            qDead.Form("SELECT idx FROM XECPMStatus WHERE id=%d AND IsBad=1;", pmStatusId);
            TSQLResult *rDead = SQLServer->Query(qDead.Data());
            if (rDead) {
              TSQLRow *rowDead;
              while ((rowDead = rDead->Next())) {
                Int_t idx = TString(rowDead->GetField(0)).Atoi();
                if (idx >= 0 && idx < kXECNChan) {
                  dead[idx] = kTRUE;
                  nDeadChannels++;
                }
                delete rowDead;
              }
              delete rDead;
            }
          }
        }
      }
    }
    std::cout << "  Dead channels: " << nDeadChannels << std::endl;

    // --- Set branch addresses ---
    MEGEventHeader *recEventHeader = new MEGEventHeader();
    TBranch *branchEventHeader = rec->GetBranch("eventheader.");
    if (branchEventHeader) branchEventHeader->SetAddress(&recEventHeader);

    TClonesArray *recXECPMCluster = new TClonesArray("MEGXECPMCluster");
    TBranch *branchXecCl = rec->GetBranch("xeccl");
    if (branchXecCl) branchXecCl->SetAddress(&recXECPMCluster);

    MEGRecData *pReco = new MEGRecData;
    TBranch *bReco = nullptr;
    rec->SetBranchAddress("reco.", &pReco, &bReco);

    TClonesArray *recBGOCEX = new TClonesArray("MEGBGOCEXResult");
    TBranch *branchBGOCEX = rec->GetBranch("bgocexresult");
    if (branchBGOCEX) branchBGOCEX->SetAddress(&recBGOCEX);

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

      // Read event header for trigger mask
      if (branchEventHeader) branchEventHeader->GetEntry(ev);
      Int_t triggermask = recEventHeader->Getmask();

      // Keep only physics triggers (50, 51); skip pedestal (63) and others
      if (triggermask != 50 && triggermask != 51) continue;

      // Read reconstruction
      if (bReco) bReco->GetEntry(ev);
      if (branchBGOCEX) branchBGOCEX->GetEntry(ev);
      if (branchXecCl) branchXecCl->GetEntry(ev);

      // --- Reconstruction quantities ---
      Float_t erec = pReco->GetEGammaAt(0);

      // Energy window: 55 MeV peak only
      if (erec < min55 || erec > max55) continue;

      // Opening angle and Etrue from 2-body kinematics
      Float_t openingangle = ((MEGBGOCEXResult*)(recBGOCEX->At(0)))->GetopeningAngle();
      Float_t sqrtarg = 0.25f * Epi0 * Epi0
                        - mpi0 * mpi0 / (2.0f * (1.0f - TMath::Cos(openingangle * TMath::Pi() / 180.0f)));
      if (sqrtarg < 0) continue;  // unphysical — skip
      Float_t etrue = Epi0 / 2.0f - TMath::Sqrt(sqrtarg);

      // --- Fill output variables ---
      run   = iRun;
      event = ev;

      energyTruth = etrue;
      energyReco  = erec;
      gstatus     = pReco->GetEvstatGammaAt(0);

      uvwRecoFI[0] = pReco->GetUGammaAt(0);
      uvwRecoFI[1] = pReco->GetVGammaAt(0);
      uvwRecoFI[2] = pReco->GetWGammaAt(0);

      Angle = openingangle;
      Ebgo  = ((MEGBGOCEXResult*)(recBGOCEX->At(0)))->GetbgoEnergy();

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

      if (recXECPMCluster->GetEntries() >= kXECNChan) {
        for (Int_t ch = 0; ch < kXECNChan; ch++) {
          MEGXECPMCluster *cl = (MEGXECPMCluster*)(recXECPMCluster->At(ch));
          npho[ch] = cl->GetnphoAt(0);
          nphe[ch] = cl->GetnpheAt(0);
          time_arr[ch] = cl->GettpmAt(0);
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

    // Cleanup
    delete recEventHeader;
    delete recXECPMCluster;
    delete pReco;
    delete recBGOCEX;
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

  delete SQLServer;
}
