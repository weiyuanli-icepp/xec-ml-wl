/*
PrepareRealDataInpainter.C
- Simplified version of PrepareRealData.C for inpainter fine-tuning.
- Reads Rec Trees (Real Data) with minimal selection (trigger mask only).
- No physics selection, no pileup cut.
- Calculates ML Features (Relative Npho, Time)
- Outputs a flat tree compatible with the Python training scripts.

Usage:
$ cd ~/meghome/offline/analyzer
$ ./meganalyzer -b -q -I 'loader.C()'
  where loader.C calls PrepareRealDataInpainterFromList(...)
*/

#include <TROOT.h>
#include <TMath.h>
#include <TChain.h>
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TSystem.h>
#include <fstream>
#include <sstream>
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
#   include "include/generated/MEGEventHeader.h"
#   include "include/generated/MEGXECPMCluster.h"
#   include "xec/xectools.h"
#   include "units/MEGSystemOfUnits.h"
#else
   class ROMETreeInfo;
   class MEGEventHeader;
   class MEGXECPMCluster;
#endif

using namespace MEG;

void _PrepareRealDataInpainter_impl(TChain *rec, TString outputFileName);

// ---------------------------------------------------------------------------
// Entry point: read one line from a run-list file.
//
//   runListFile  Text file with two columns: <run_number> <file_path>
//   jobIndex     0-based line index (maps to SLURM_ARRAY_TASK_ID)
//   outputDir    Directory for the output ROOT file
// ---------------------------------------------------------------------------
void PrepareRealDataInpainterFromList(TString runListFile, Int_t jobIndex, TString outputDir = ".")
{
   std::ifstream ifs(runListFile.Data());
   if (!ifs.is_open()) {
      std::cout << "[Error] Cannot open run list: " << runListFile << std::endl;
      return;
   }

   std::string line;
   Int_t idx = 0;
   Int_t runNumber = -1;
   std::string filePath;
   while (std::getline(ifs, line)) {
      if (line.empty() || line[0] == '#') { continue; }
      if (idx == jobIndex) {
         std::istringstream iss(line);
         iss >> runNumber >> filePath;
         break;
      }
      idx++;
   }
   ifs.close();

   if (runNumber < 0 || filePath.empty()) {
      std::cout << "[Error] Job index " << jobIndex << " out of range or malformed line" << std::endl;
      return;
   }

   if (gSystem->AccessPathName(filePath.c_str())) {
      std::cout << "[Error] File not found: " << filePath << std::endl;
      return;
   }

   TChain *rec = new TChain("rec");
   rec->Add(filePath.c_str());
   std::cout << "[INFO] Processing run " << runNumber << ": " << filePath << std::endl;

   gSystem->mkdir(outputDir.Data(), kTRUE);
   TString outFile = Form("%s/DataGammaAngle_%06d.root", outputDir.Data(), runNumber);
   _PrepareRealDataInpainter_impl(rec, outFile);
}

// ---------------------------------------------------------------------------
// Entry point: read a range of lines from a run-list file and chain them
// into a single output file. Used for combining multiple runs (e.g. one day)
// into one ROOT file.
//
//   runListFile  Text file with two columns: <run_number> <file_path>
//   startIdx     0-based starting line index (non-comment lines)
//   nFiles       Number of lines to read from startIdx
//   outputDir    Directory for the output ROOT file
//   dateLabel    Label for the output filename (e.g. "2023-11-15")
// ---------------------------------------------------------------------------
void PrepareRealDataInpainterFromListRange(TString runListFile, Int_t startIdx, Int_t nFiles,
                                            TString outputDir = ".", TString dateLabel = "")
{
   std::ifstream ifs(runListFile.Data());
   if (!ifs.is_open()) {
      std::cout << "[Error] Cannot open run list: " << runListFile << std::endl;
      return;
   }

   TChain *rec = new TChain("rec");
   std::string line;
   Int_t idx = 0;
   Int_t nAdded = 0;
   Int_t firstRun = -1;

   while (std::getline(ifs, line)) {
      if (line.empty() || line[0] == '#') { continue; }
      if (idx >= startIdx && idx < startIdx + nFiles) {
         Int_t runNumber = -1;
         std::string filePath;
         std::istringstream iss(line);
         iss >> runNumber >> filePath;

         if (runNumber < 0 || filePath.empty()) {
            std::cout << "[WARN] Malformed line at index " << idx << ", skipping" << std::endl;
         } else if (gSystem->AccessPathName(filePath.c_str())) {
            std::cout << "[WARN] File not found: " << filePath << ", skipping" << std::endl;
         } else {
            rec->Add(filePath.c_str());
            std::cout << "[INFO] Added run " << runNumber << ": " << filePath << std::endl;
            if (firstRun < 0) firstRun = runNumber;
            nAdded++;
         }
      }
      if (idx >= startIdx + nFiles) break;
      idx++;
   }
   ifs.close();

   if (nAdded == 0) {
      std::cout << "[Error] No valid files found in range [" << startIdx << ", "
                << startIdx + nFiles << ")" << std::endl;
      return;
   }

   std::cout << "[INFO] Chained " << nAdded << " files for processing" << std::endl;

   gSystem->mkdir(outputDir.Data(), kTRUE);
   TString outFile;
   if (dateLabel.Length() > 0) {
      outFile = Form("%s/DataGammaAngle_%s.root", outputDir.Data(), dateLabel.Data());
   } else {
      outFile = Form("%s/DataGammaAngle_%06d.root", outputDir.Data(), firstRun);
   }
   _PrepareRealDataInpainter_impl(rec, outFile);
}

// ---------------------------------------------------------------------------
// Default entry point (not typically used with runlist workflow)
// ---------------------------------------------------------------------------
void PrepareRealDataInpainter(Int_t sRun = 430000, Int_t nfile = 100)
{
   std::cout << "[INFO] Use PrepareRealDataInpainterFromList for runlist-based processing." << std::endl;
   std::cout << "[INFO] This entry point is a placeholder." << std::endl;
}

// ---------------------------------------------------------------------------
// Implementation: minimal selection, extract npho/time arrays.
// ---------------------------------------------------------------------------
void _PrepareRealDataInpainter_impl(TChain *rec, TString outputFileName)
{
   // =========================================================================
   // OUTPUT SETUP
   // =========================================================================
   TFile outputFile(outputFileName, "RECREATE", "Real Data ML Input (Simple)");
   TTree *tree = new TTree("tree", "Real Data Inference Tree");

   static const int kXECNChan = 4760;

   Int_t   run_out = 0;
   Int_t   event_out = 0;

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
   tree->Branch("npho", npho, Form("npho[%d]/F", kXECNChan));
   tree->Branch("time", time, Form("time[%d]/F", kXECNChan));
   tree->Branch("relative_npho", relative_npho, Form("relative_npho[%d]/F", kXECNChan));
   tree->Branch("relative_time", relative_time, Form("relative_time[%d]/F", kXECNChan));
   tree->Branch("ch_npho_max", &ch_npho_max, "ch_npho_max/S");
   tree->Branch("ch_time_min", &ch_time_min, "ch_time_min/S");
   tree->Branch("npho_max_used", &npho_max_used, "npho_max_used/F");
   tree->Branch("time_min_used", &time_min_used, "time_min_used/F");

   // =========================================================================
   // READERS
   // =========================================================================
   TTreeReader reader(rec);

   TTreeReaderValue<ROMETreeInfo>   infoRV(reader, "Info.");
   TTreeReaderValue<MEGEventHeader> headerRV(reader, "eventheader.");
   TTreeReaderArray<MEGXECPMCluster> xecclRA(reader, "xeccl");

   // =========================================================================
   // EVENT LOOP
   // =========================================================================
   Int_t nTotal = rec->GetEntries();
   Int_t nProcessed = 0;
   Int_t nSelected = 0;

   std::cout << "Starting loop over " << nTotal << " events..." << std::endl;

   while (reader.Next()) {
      if (nProcessed % 1000 == 0) std::cout << "Processing " << nProcessed << " / " << nTotal << std::endl;
      nProcessed++;

      // --- Trigger Selection ---
      if (headerRV->Getmask() != 0) continue;

      // --- Data Extraction ---
      run_out   = infoRV->GetRunNumber();
      event_out = infoRV->GetEventNumber();

      // Initialize arrays
      for (int ch = 0; ch < kXECNChan; ch++) {
         npho[ch] = 1e10f;
         nphe[ch] = 1e10f;
         time[ch] = 1e10f;
         relative_npho[ch] = 1e10f;
         relative_time[ch] = 1e10f;
      }

      if (xecclRA.GetSize() != kXECNChan) continue;

      for (int ch = 0; ch < kXECNChan; ++ch) {
         MEGXECPMCluster &cl = xecclRA.At(ch);
         npho[ch] = cl.GetnphoAt(0);
         nphe[ch] = cl.GetnpheAt(0);
         time[ch] = cl.GettpmAt(0);
      }

      // --- Calculate Relative Arrays ---
      Float_t val_max  = -std::numeric_limits<Float_t>::infinity();
      Float_t min_time =  std::numeric_limits<Float_t>::infinity();

      ch_npho_max = -1;
      ch_time_min = -1;
      npho_max_used = 1e10f;
      time_min_used = 1e10f;

      for (Int_t ch = 0; ch < kXECNChan; ++ch) {
         const Float_t v = npho[ch];
         const Float_t n = nphe[ch];
         if (std::isfinite(v) && v >= 0.0f && v < 1e9f) {
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

      for (Int_t ch = 0; ch < kXECNChan; ++ch) {
         if (ch_npho_max >= 0 && std::isfinite(val_max) && std::fabs(val_max) > 0) {
            relative_npho[ch] = (std::isfinite(npho[ch]) && npho[ch] < 1e9f) ? (npho[ch] / val_max) : 1e10f;
         }

         if (std::isfinite(min_time) && min_time < 1e9f && std::isfinite(time[ch]) && time[ch] < 1e9f) {
            relative_time[ch] = time[ch] - min_time;
         }
      }

      tree->Fill();
      nSelected++;
   }

   tree->Write();
   outputFile.Close();

   std::cout << "Done! Selected " << nSelected << " / " << nProcessed << " events." << std::endl;
   std::cout << "Output: " << outputFile.GetName() << std::endl;
}
