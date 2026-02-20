#include <TChain.h>
#include <TString.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TVector3.h>
#include <ROMETreeInfo.h>
#include <Riostream.h>
#include <fstream>
#include <limits>
#include <cmath>
#include <algorithm>

#include "/data/user/ext-li_w1/meghome/offline/analyzer/include/generated/MEGAllFolders.h"
#include "/data/user/ext-li_w1/meghome/offline/common/include/xec/xectools.h"

// --- Exact solid angle computation (from PMSolidAngle.cpp in the official analyzer) ---
namespace {

// SiPM/PMT channel boundary: channels 0..4091 = SiPM, 4092..4759 = PMT
// (kMaxNXePM - kMaxNXePMT = 4760 - 668 = 4092)
const Int_t kSiPMPMTBoundary = 4092;

// PMT photocathode radius [cm] (from xecconst.h: kRCATH = 4.5/2 * centimeter)
const Double_t kPMTCathodeRadius = 2.25;

// Exact solid angle for MPPC (SiPM): 2x2 chip rectangular geometry.
// Replicates PMSolidAngle::PMSolidAngleMPPC from the official analyzer.
// Returns fractional solid angle (omega / 4pi).
Double_t ComputeSolidAngleMPPC(TVector3 view, TVector3 center, TVector3 normal)
{
   TVector3 center_view = center - view;
   if (center_view.Dot(normal) > 0) {
      return 0;
   }

   // Set up local coordinate system (U, V, W=normal)
   TVector3 unit[3];
   unit[0].SetXYZ(0, 0, 1); // U direction
   unit[1] = (unit[0].Cross(normal)).Unit(); // V direction
   unit[2] = normal.Unit(); // W direction
   unit[0] = (unit[1].Cross(unit[2])).Unit();

   // MPPC chip geometry: 4 chips of 5.90 mm separated by 0.5 mm gap
   const Double_t ChipDistance = 0.05;  // 0.5 mm [cm]
   const Double_t ChipSize    = 0.59;   // 5.90 mm [cm]
   Double_t sin_a1, sin_a2, sin_b1, sin_b2;
   Double_t solid_total = 0;

   // Chip 1: (+U, +V) quadrant
   TVector3 vcorner1 = center + ChipDistance / 2.0 * unit[0] + ChipDistance / 2.0 * unit[1];
   TVector3 vcorner2 = vcorner1 + ChipSize * unit[0] + ChipSize * unit[1];
   TVector3 v1       = vcorner1 - view;
   TVector3 v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total += TMath::Abs(asin(sin_a1 * sin_b1) +
                             asin(sin_a2 * sin_b2) -
                             asin(sin_a1 * sin_b2) -
                             asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   // Chip 2: (-U, +V) quadrant
   vcorner1 = center - ChipDistance / 2.0 * unit[0] + ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 - ChipSize * unit[0] + ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total += TMath::Abs(asin(sin_a1 * sin_b1) +
                             asin(sin_a2 * sin_b2) -
                             asin(sin_a1 * sin_b2) -
                             asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   // Chip 3: (+U, -V) quadrant
   vcorner1 = center   + ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 + ChipSize        * unit[0] - ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total += TMath::Abs(asin(sin_a1 * sin_b1) +
                             asin(sin_a2 * sin_b2) -
                             asin(sin_a1 * sin_b2) -
                             asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   // Chip 4: (-U, -V) quadrant
   vcorner1 = center   - ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 - ChipSize        * unit[0] - ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total += TMath::Abs(asin(sin_a1 * sin_b1) +
                             asin(sin_a2 * sin_b2) -
                             asin(sin_a1 * sin_b2) -
                             asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   return solid_total;
}

// Exact solid angle for PMT: Paxton formula for circular disk.
// Replicates PMSolidAngle::PMSolidAnglePMT from the official analyzer.
// Returns fractional solid angle (omega / 4pi).
Double_t ComputeSolidAnglePMT(TVector3 view, TVector3 center, TVector3 normal)
{
   TVector3 center_view = view - center;
   if (center_view.Dot(normal) <= 0) {
      return 0;
   }

   Double_t solid_angle = 0;
   const Double_t Rm = kPMTCathodeRadius;

   Double_t dist  = center_view.Mag();
   Double_t L     = center_view.Dot(normal);
   Double_t Theta = center_view.Angle(normal);
   Double_t R0    = dist * TMath::Sin(Theta);

   Double_t Rmax    = TMath::Sqrt(L * L + (R0 + Rm) * (R0 + Rm));
   Double_t R1      = TMath::Sqrt(L * L + (R0 - Rm) * (R0 - Rm));
   Double_t kappa   = TMath::Sqrt(1 - R1 * R1 / Rmax / Rmax);
   Double_t alphasq = 4 * R0 * Rm / (R0 + Rm) / (R0 + Rm);

   if (TMath::Abs((R0 - Rm) / Rm) < 1e-03) {
      solid_angle = TMath::Pi() -
                    2 * L / Rmax * std::comp_ellint_1(kappa);
   } else if (R0 < Rm) {
      solid_angle = 2 * TMath::Pi() -
                    2 * L / Rmax * std::comp_ellint_1(kappa) +
                    2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(kappa, alphasq);
   } else if (R0 > Rm) {
      solid_angle =
         - 2 * L / Rmax * std::comp_ellint_1(kappa) +
         2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(kappa, alphasq);
   } else {
      return 0;
   }

   return solid_angle / (4 * TMath::Pi());
}

} // anonymous namespace

// Use submit_SingleRun.py to submit batch jobs for multiple single runs

// runOffset: adds offset to output run number and filename (useful for combining datasets)
//   e.g., runOffset=200 with iStart=0 outputs MCGamma_200.root with run=200
// void MCXECPreprocess(Int_t iStart=0, Int_t iEnd=0, TString conf="E52.8_AngUni_PosSQ", TString outputDir="../../xec-ml-wl/data/E52.8_AngUni_PosSQ/single_run", Int_t ActivateRandomization=0, Int_t RandomSeed=42, Int_t runOffset=0)
// void MCXECPreprocess(Int_t iStart=0, Int_t iEnd=0, TString conf="E35to55_AngUni_PosSQ", TString outputDir="../../xec-ml-wl/data/E35to55_AngUni_PosSQ/single_run", Int_t ActivateRandomization=0, Int_t RandomSeed=42, Int_t runOffset=0)
void MCXECPreprocess(Int_t iStart=0, Int_t iEnd=0, TString conf="E15to60_AngUni_PosSQ", TString outputDir="../../xec-ml-wl/data/E15to60_AngUni_PosSQ/single_run", Int_t ActivateRandomization=0, Int_t RandomSeed=42, Int_t runOffset=0)
{
   XECTOOLS::InitXECGeometryParameters(XECTOOLS::UVWDefinition::Global,
                                       64.84, 106.27, 67.03, 96., 125.52,
                                       0, 0, 0);

   // --- Input chains ---
   TChain *rec = new TChain("rec");
   TChain *sim = new TChain("sim");

    TString commonDir = gSystem->ExpandPathName("$mcxec");
    TString recBaseDirectory = Form("%s/anaOut/%s", commonDir.Data(), conf.Data());
    for (Int_t iRun=iStart; iRun<=iEnd; iRun++) rec->Add(Form("%s/rec%05d.root", recBaseDirectory.Data(), iRun));
    TString simBaseDirectory = Form("%s/barOut/%s", commonDir.Data(), conf.Data());
    for (Int_t iRun=iStart; iRun<=iEnd; iRun++) sim->Add(Form("%s/sim%05d.root", simBaseDirectory.Data(), iRun));

   // --- Output file & trees ---
   // Apply runOffset to output run numbers
   const Int_t outStart = iStart + runOffset;
   const Int_t outEnd   = iEnd + runOffset;
   if (runOffset != 0) {
     std::cout << "Run offset: " << runOffset << " (input " << iStart << "-" << iEnd
               << " -> output " << outStart << "-" << outEnd << ")\n";
   }

   TString baseName;
   if (ActivateRandomization) {
     std::cout << "Randomization activated with seed " << RandomSeed << "\n";
     if (iStart == iEnd) {
       baseName = Form("MCGamma_Randomized_seed%d_%d", RandomSeed, outStart);
     } else {
       baseName = Form("MCGamma_Randomized_seed%d_%d-%d", RandomSeed, outStart, outEnd);
     }
   } else {
     std::cout << "Randomization not activated\n";
     if (iStart == iEnd) {
       baseName = Form("MCGamma_%d", outStart);
     } else {
       baseName = Form("MCGamma_%d-%d", outStart, outEnd);
     }
   }
   TFile outputFile(Form("%s/%s.root", outputDir.Data(), baseName.Data()),
                    "RECREATE", "XEC MC single-candidate");
   TTree *tree = new TTree("tree", "Tree for Gamma variables prediction ML training samples");

   static const int kXECNChan = 4760;

   Int_t   run=0;
   Int_t   event=0;

   Float_t xyzRecoFI[3];  // first interaction point reconstructed
   Float_t xyzRecoLC[3];  // light center reconstructed
   Float_t xyzTruth[3];   // first interaction point truth
   Float_t xyzVTX[3];     // photon starting point
   Float_t emiVec[3];     // photon emission vector

   Float_t uvwRecoFI[3];  // first interaction point reconstructed
   Float_t uvwRecoLC[3];  // light center reconstructed
   Float_t uvwTruth[3];   // first interaction point truth
   Float_t emiAng[2];     // photon emission angle (theta, phi)

   Float_t energyReco  = 1e10f;  // reconstructed energy
   Float_t energyTruth = 1e10f;  // energy truth
   Float_t timeReco    = 1e10f;  // reconstructed first interaction time
   Float_t timeTruth   = 1e10f;  // first interaction time truth

   Float_t npho[kXECNChan];           // npho for each channel
   Float_t nphe[kXECNChan];           // nphe for each channel
   Float_t time[kXECNChan];           // time for each channel
   Float_t relative_npho[kXECNChan];  // npho[i] / npho[largest]  for each channel
   Float_t relative_time[kXECNChan];  // time[i] - time[earliest] for each channel
   Float_t solid_angle[kXECNChan];    // fractional solid angle per channel (omega_i / 4pi)

   Short_t ch_npho_max = -1, ch_time_min = -1;
   Float_t npho_max_used = 1e10f, time_min_used = 1e10f;

   tree->Branch("run",   &run,   "run/I"); // commented out
   tree->Branch("event", &event, "event/I"); // commented out

   tree->Branch("xyzRecoFI", xyzRecoFI, "xyzRecoFI[3]/F");
   tree->Branch("xyzRecoLC", xyzRecoLC, "xyzRecoLC[3]/F");
   tree->Branch("xyzTruth", xyzTruth, "xyzTruth[3]/F"); // commented out
   tree->Branch("xyzVTX",   xyzVTX,   "xyzVTX[3]/F");
   tree->Branch("emiVec",   emiVec,   "emiVec[3]/F"); // commented out

   tree->Branch("uvwRecoFI", uvwRecoFI, "uvwRecoFI[3]/F");
   tree->Branch("uvwRecoLC", uvwRecoLC, "uvwRecoLC[3]/F");
   tree->Branch("uvwTruth", uvwTruth, "uvwTruth[3]/F"); // commented out
   tree->Branch("emiAng",   emiAng,   "emiAng[2]/F"); // commented out

   tree->Branch("energyReco",  &energyReco,  "energyReco/F");
   tree->Branch("energyTruth", &energyTruth, "energyTruth/F"); // commented out
   tree->Branch("timeReco",    &timeReco,    "timeReco/F");
   tree->Branch("timeTruth",   &timeTruth,   "timeTruth/F"); // commented out

   tree->Branch("npho", npho, Form("npho[%d]/F", kXECNChan));
   tree->Branch("nphe", nphe, Form("nphe[%d]/F", kXECNChan));
   tree->Branch("time", time, Form("time[%d]/F", kXECNChan));
   tree->Branch("relative_npho", relative_npho, Form("relative_npho[%d]/F", kXECNChan)); // commented out
   tree->Branch("relative_time", relative_time, Form("relative_time[%d]/F", kXECNChan)); // commented out

   tree->Branch("ch_npho_max", &ch_npho_max, "ch_npho_max/S");
   tree->Branch("ch_time_min", &ch_time_min, "ch_time_min/S");

   tree->Branch("npho_max_used", &npho_max_used, "npho_max_used/F");
   tree->Branch("time_min_used", &time_min_used, "time_min_used/F");
   tree->Branch("solid_angle", solid_angle, Form("solid_angle[%d]/F", kXECNChan));

   // --- Readers (rec) ---
   TTreeReader reader(rec);
   TTreeReaderValue<ROMETreeInfo>                infoRV(reader, "Info.");               // run, event
   TTreeReaderValue<MEGEventHeader>              eventheaderRV(reader, "eventheader.");
   TTreeReaderArray<MEGXECPosLocalFitResult>     xecposfitRA(reader, "xecposlfit");     // xyz
   TTreeReaderArray<MEGXECTimeFitResult>         xectimefitRA(reader, "xectimefit");    // timeReco
   TTreeReaderArray<MEGXECEneTotalSumRecResult>  xecenerecRA(reader, "xecenetotalsum"); // energyReco
   TTreeReaderArray<MEGXECPMCluster>             xecclRA(reader, "xeccl");              // npho, time

   // --- Readers (sim) ---
   TTreeReader simreader(sim);
   TTreeReaderValue<MEGMCTriggerEvent> mctriggerRV(simreader, "mctrigger.");
   TTreeReaderValue<MEGMCMixtureInfoEvent> mcmixRV(simreader, "mcmixevent.");
   TTreeReaderArray<MEGMCXECHit>   mcxechitRA(simreader, "mcxechit");
   TTreeReaderValue<MEGMCKineEvent> mckineRV(simreader, "mckine.");

   // --- Event loop ---
   const Int_t nTotalEvents = rec->GetEntries();
   Int_t nProcessed = 0;

   while (reader.Next()) {
     simreader.Next();
     ++nProcessed;
     if ((nProcessed-1) % 1000 == 0)
       std::cout << (nProcessed-1) << "/" << nTotalEvents << " events read\n";

     run   = infoRV->GetRunNumber() + runOffset;
     event = infoRV->GetEventNumber();

     for (int i=0;i<3;i++) {
       xyzRecoFI[i] = 1e10f; uvwRecoFI[i] = 1e10f;
       xyzRecoLC[i] = 1e10f; uvwRecoLC[i] = 1e10f;
       xyzTruth[i]  = 1e10f; uvwTruth[i]  = 1e10f;
       xyzVTX[i]    = 1e10f;
       emiVec[i]    = 1e10f;
     }
     emiAng[0] = 1e10f; emiAng[1] = 1e10f;
     energyReco     = 1e10f;
     energyTruth    = 1e10f;
     timeReco       = 1e10f;
     timeTruth      = 1e10f;

     for (int ch=0; ch<kXECNChan; ++ch) {
       npho[ch] = 1e10f;
       nphe[ch] = 1e10f;
       time[ch] = 1e10f;
       relative_npho[ch] = 1e10f;
       relative_time[ch] = 1e10f;
       solid_angle[ch] = 0.0f;
     }

     const int nRecoGamma = xectimefitRA.GetSize();
     if (nRecoGamma != 1 || xecposfitRA.GetSize() < 1 || xecenerecRA.GetSize() < 1) {
       continue;
     }

     // --- Find the primary with the highest energy in SIM (sevID, primaryID) ---
     Int_t sevID = -1, primaryID = -1;
     Double_t maxE = -1.0;

     const int nPrimary = mckineRV->Getnprimary();
     for (int i=0; i<nPrimary; ++i) {
        const Double_t px = mckineRV->GetxmomAt(i);
        const Double_t py = mckineRV->GetymomAt(i);
        const Double_t pz = mckineRV->GetzmomAt(i);
        const Double_t e  = TMath::Sqrt(px*px + py*py + pz*pz);

        //  if (mckineRV->GettypeAt(i) != 22) continue; // only gamma
        if (e > maxE) {
          maxE = e;
          sevID = mckineRV->GetsevidAt(i);
          primaryID = i;
        }
     }

     if (sevID < 0 || primaryID < 0) {
       continue;
     }

     // --- mcmix index for sevID
     Int_t mcmixID = -1;
     for (int isev=0; isev<mcmixRV->Getnsev(); ++isev) {
       if (mcmixRV->GetSevIDAt(isev) == sevID) { mcmixID = isev; break; }
     }
     if (mcmixID < 0) {
       continue;
     }

     // --- mcxechit index for sevID
     Int_t mcxechitID = -1;
     Int_t idx = -1;
     for (auto&& h : mcxechitRA) {
       ++idx;
       if (h.Getsevid() == sevID) {
         mcxechitID = idx;
         break;
       }
     }
     if (mcxechitID < 0) {
       continue;
     }

     // --- Electronics and trigger offsets (ns)
     const Double_t electronicsOffset = 1e9 * mctriggerRV->GetDRSSamplingPhaseShift();
     const Double_t triggerOffset     = 1e9 * mcmixRV->GetOffsetAt(mcmixID);

     // --- Fill TRUTH ---
     {
       const auto& mcHit = mcxechitRA.At(mcxechitID);
       energyTruth = mcHit.Getenergy();                   // energy
       if (energyTruth < 1.45e-2) continue; // skip events with energy less than 14.5 MeV
       timeTruth   = electronicsOffset + triggerOffset;   // time

       xyzVTX[0] = mckineRV->GetxvtxAt(primaryID);        // emission position
       xyzVTX[1] = mckineRV->GetyvtxAt(primaryID);
       xyzVTX[2] = mckineRV->GetzvtxAt(primaryID);

       Double_t xyzT[3];
       Double_t uvwT[3];
       for (int i=0;i<3;i++) xyzT[i] = mcHit.GetxyzAt(i);
       XECTOOLS::XYZ2UVW(xyzT, uvwT);
       for (int i=0;i<3;i++) uvwTruth[i] = static_cast<Float_t>(uvwT[i]);

       for (int i=0;i<3;i++) {
         xyzTruth[i] = xyzT[i];                           // first interaction position
         emiVec[i]   = xyzT[i] - xyzVTX[i];               // emission vector
       }
       TVector3 emiTVec(emiVec);
       if (emiTVec.Mag() > 0) {
         emiTVec = emiTVec.Unit();
       }
       for (int i=0;i<3;i++) emiVec[i] = emiTVec[i];
       emiAng[0] = emiTVec.Theta()*180.0/TMath::Pi();
       // emiAng[1] = emiTVec.Phi()*180.0/TMath::Pi();
       emiAng[1] = TMath::ATan2(emiTVec[1], -emiTVec[0]) * 180.0 / TMath::Pi();
     }

     // --- Reconstruction ---
     const MEGXECTimeFitResult&        aTime = xectimefitRA.At(0);
     const MEGXECPosLocalFitResult&    aPos  = xecposfitRA.At(0);
     const MEGXECEneTotalSumRecResult& aEne  = xecenerecRA.At(0);

     energyReco = aEne.Getenergy();

     for (int i=0;i<3;i++) {
       xyzRecoFI[i] = aPos.GetxyzAt(i);
       uvwRecoFI[i] = aPos.GetuvwAt(i);
     }

     Double_t uvwLC_d[3];
     Double_t xyzLC_d[3];
     for (int i=0;i<3;i++) {
       uvwLC_d[i]   = aPos.GetuvwLightCenterAt(i);
       uvwRecoLC[i] = aPos.GetuvwLightCenterAt(i);
     }
     XECTOOLS::UVW2XYZ(uvwLC_d, xyzLC_d);
     for (int i=0;i<3;i++) xyzRecoLC[i] = static_cast<Float_t>(xyzLC_d[i]);

     {
       TVector3 xyzRecV(aPos.GetxyzAt(0), aPos.GetxyzAt(1), aPos.GetxyzAt(2));
       TVector3 xyzEmV(xyzVTX);
       const TVector3 flight = xyzRecV - xyzEmV;
       timeReco = 1e9 * aTime.Gettime() - flight.Mag()/29.9792458; // ns
     }

     // --- npho and time from xeccl ---
     {
       if (xecclRA.GetSize() == kXECNChan) {
         for (int ch=0; ch<kXECNChan; ++ch) {
           MEGXECPMCluster &cl = xecclRA.At(ch);
           // randomize npho and nphe uniformly for +-5% if activated
           if (ActivateRandomization) {
             Float_t npho_orig = cl.GetnphoAt(0);
             Float_t nphe_orig = cl.GetnpheAt(0);
             if (npho_orig > 0) {
                TRandom3 randGen(RandomSeed + run + event + ch);
                Float_t npho_rnd = randGen.Uniform(0.95f*npho_orig, 1.05f*npho_orig);
                Float_t nphe_rnd = randGen.Uniform(0.95f*nphe_orig, 1.05f*nphe_orig);
                npho[ch] = npho_rnd;
                nphe[ch] = nphe_rnd;
             } else {
               npho[ch] = npho_orig;
               nphe[ch] = nphe_orig;
             }
           } else {
             npho[ch] = cl.GetnphoAt(0);
             nphe[ch] = cl.GetnpheAt(0);
           }
           time[ch] = cl.GettpmAt(0);
         }
       }
     }

     // --- Solid angle computation ---
     // Compute fractional solid angle (omega_i / 4pi) for each PM relative to xyzRecoFI.
     // Uses exact methods from PMSolidAngle (official analyzer):
     //   SiPM (ch 0-4091):  ComputeSolidAngleMPPC (4-chip rectangular geometry)
     //   PMT  (ch 4092-4759): ComputeSolidAnglePMT (Paxton circular disk formula)
     if (std::isfinite(xyzRecoFI[0]) && xyzRecoFI[0] < 1e9f) {
       TVector3 vtx(xyzRecoFI[0], xyzRecoFI[1], xyzRecoFI[2]);
       for (int iPM = 0; iPM < kXECNChan; ++iPM) {
         Double_t pmXYZ[3], pmDir[3];
         Bool_t gotGeom = kFALSE;

         Int_t face = -1, row = -1, col = -1;
         if (XECTOOLS::GetFaceRowColFromCh(iPM, face, row, col)) {
           if (XECTOOLS::GetXYZFromFaceRowCol(face, row, col, pmXYZ) &&
               XECTOOLS::GetDirectionFromFaceRowCol(face, row, col, pmDir)) {
             gotGeom = kTRUE;
           }
         }

         if (gotGeom) {
           TVector3 pmPos(pmXYZ[0], pmXYZ[1], pmXYZ[2]);
           TVector3 pmNorm(pmDir[0], pmDir[1], pmDir[2]);
           if (iPM < kSiPMPMTBoundary) {
             solid_angle[iPM] = static_cast<Float_t>(ComputeSolidAngleMPPC(vtx, pmPos, pmNorm));
           } else {
             solid_angle[iPM] = static_cast<Float_t>(ComputeSolidAnglePMT(vtx, pmPos, pmNorm));
           }
         }
       }
     }

     // --- relative npho and time ---
     Float_t val_max  = -std::numeric_limits<Float_t>::infinity();
     Float_t min_time =  std::numeric_limits<Float_t>::infinity();

     // Find max npho and min time
     ch_npho_max = -1;
     ch_time_min = -1;
     npho_max_used = 1e10f;
     time_min_used = 1e10f;
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
       if (std::isfinite(t) && t < min_time && n>50) {
         min_time = t;
         ch_time_min = ch;
       }
     }
     npho_max_used = (std::isfinite(val_max) && val_max < 1e9f) ? val_max : 1e10f;
     time_min_used = (std::isfinite(min_time) && min_time < 1e9f) ? min_time : 1e10f;
     if (npho_max_used == 1e10f || time_min_used == 1e10f) continue;

     // Fill relatives
     for (Int_t ch=0; ch<kXECNChan; ++ch) {
       if (ch_npho_max >= 0 && std::isfinite(val_max) && std::fabs(val_max) > 0) {
         relative_npho[ch] = (std::isfinite(npho[ch]) && npho[ch] < 1e9f) ? (npho[ch] / val_max) : 1e10f;
       } else {
         relative_npho[ch] = 1e10f;
       }

       if (std::isfinite(min_time) && min_time < 1e9f && std::isfinite(time[ch]) && time[ch] < 1e9f) {
          relative_time[ch] = time[ch] - min_time;
         // randomize relative time uniformly for +-5% if activated
          if (ActivateRandomization) {
            TRandom3 randGen(RandomSeed + run + event + ch + 10000);
            Float_t time_rnd = randGen.Uniform(0.95f*relative_time[ch], 1.05f*relative_time[ch]);
            relative_time[ch] = time_rnd;
          }
       } else {
         relative_time[ch] = 1e10f;
       }
     }
     tree->Fill();
   }
   tree->Write();
   outputFile.Close();
   std::cout << "Wrote " << outputFile.GetName() << "\n";
}
