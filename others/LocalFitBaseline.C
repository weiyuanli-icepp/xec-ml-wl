// LocalFitBaseline.C — Standalone ROOT macro for local fit position reconstruction
// with dead channel exclusion, and dead channel npho prediction.
//
// Reimplements Stages 1 & 2 of MEGTXECPosLocalFit while skipping dead channels,
// then predicts npho for dead channels as physics-based baseline for ML comparison.
//
// Usage:
//   root -l -b -q 'others/LocalFitBaseline.C("path/to/MCGamma_run.root", "dead_channels.txt")'
//   root -l -b -q 'others/LocalFitBaseline.C("path/to/MCGamma_run.root", "dead_channels.txt", "output.root")'
//
// The optional third argument specifies a custom output file path.

#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TMath.h>
#include <TMinuit.h>
#include <TVector3.h>
#include <TH1D.h>
#include <TF1.h>

#include <fstream>
#include <iostream>
#include <set>
#include <cmath>

// ========================================================================
//  Constants (from xecconst.h, xectools.h, MEGTXECPosLocalFit.cpp)
// ========================================================================

static const Int_t    kXECNChan   = 4760;
static const Int_t    kNInner     = 4092;       // inner face SiPMs: channels 0-4091
static const Int_t    kNCols      = 44;         // kNUMPM1[0]
static const Int_t    kNRows      = 93;         // kNUMPM2[0]
static const Double_t kSiPMSize   = 1.2;        // cm, SiPM active area side length
static const Double_t kXERIN      = 64.84;      // cm, inner radius of spacer
static const Double_t kMPPCHeight = 0.13;       // cm, 1.3 mm offset from spacer to photoelectric surface
static const Double_t kReff       = kXERIN + kMPPCHeight; // 64.97 cm, effective radius
static const Double_t kPhiSiPM   = TMath::ASin(kSiPMSize / kXERIN);
static const Double_t kAtten     = 1e6;         // cm, effectively infinite attenuation length

// gXEPHI_rad = 125.52 * pi/180
static const Double_t gXEPHI_rad = 125.52 * TMath::DegToRad();

// PM intervals: spacing between adjacent SiPMs in UVW coordinates (cm).
// The official code computes these from the run header. For the standard MC geometry,
// the 44 columns along U span ~66.4 cm with spacing ~1.51 cm, and the 93 rows along V
// are spaced by gXEPHI_rad * kReff / kNRows.

// ========================================================================
//  Geometry Parameters
// ========================================================================

// These will be set in the macro body
static Double_t gPMIntervalU = 0;
static Double_t gPMIntervalV = 0;

// PM geometry caches (analytical or from run header)
static Double_t gPMPos[kXECNChan][3];
static Double_t gPMDir[kXECNChan][3];
static Double_t gPMU[kNInner], gPMV[kNInner]; // UVW coordinates of inner SiPMs

// Dead channel set
static std::set<Int_t> gDeadSet;

// QE values for chi-square weighting
static const Double_t kQE_SiPM = 0.12;
static const Double_t kQE_PMT  = 0.16;

// Global variables for FCNSolid (TMinuit callback interface)
static Float_t  gNphoArr[kNInner];
static Bool_t   gPMUsed[kNInner]; // which PMs to use in Stage 2 fit
static Int_t    gNDF = 0;

// ========================================================================
//  Coordinate Transforms (standalone, Global UVW definition, no ad-hoc shift)
// ========================================================================

void MyUVW2XYZ(const Double_t *uvw, Double_t *xyz)
{
   // From xectools.cpp:532-537 (Global definition, no ad-hoc shift)
   xyz[0] = -1.0 * (uvw[2] + kReff) * TMath::Cos(uvw[1] / kReff);
   xyz[1] =        (uvw[2] + kReff) * TMath::Sin(uvw[1] / kReff);
   xyz[2] = uvw[0];
}

void MyXYZ2UVW(const Double_t *xyz, Double_t *uvw)
{
   // From xectools.h:240-289 (Global definition, no ad-hoc shift)
   Double_t r   = TMath::Sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
   Double_t phi = (xyz[0] == 0.0 && xyz[1] == 0.0)
                  ? 0.0
                  : TMath::Pi() + TMath::ATan2(-xyz[1], -xyz[0]); // in [0, 2pi]
   uvw[0] = xyz[2];                                // U = z
   uvw[1] = -(phi - TMath::Pi()) * kReff;          // V = -(phi - pi) * Reff
   uvw[2] = r - kReff;                             // W = r - Reff
}

TVector3 UVW2XYZVec(Double_t u, Double_t v, Double_t w)
{
   Double_t uvw[3] = {u, v, w};
   Double_t xyz[3];
   MyUVW2XYZ(uvw, xyz);
   return TVector3(xyz[0], xyz[1], xyz[2]);
}

// ========================================================================
//  Solid Angle: MPPC (SiPM) — verbatim from PMSolidAngle.cpp:78-160
// ========================================================================

Double_t PMSolidAngleMPPC(TVector3 view, TVector3 center, TVector3 normal)
{
   TVector3 center_view = center - view;
   if (center_view.Dot(normal) > 0) {
      return 0;
   }

   TVector3 unit[3];
   unit[0].SetXYZ(0, 0, 1); // U direction
   unit[1] = (unit[0].Cross(normal)).Unit(); // V direction
   unit[2] = normal.Unit(); // W direction
   unit[0] = (unit[1].Cross(unit[2])).Unit();

   const Double_t ChipDistance = 0.05;  // 0.5 mm [cm]
   const Double_t ChipSize    = 0.59;   // 5.90 mm [cm]
   Double_t sin_a1, sin_a2, sin_b1, sin_b2;
   Double_t solid_total = 0;

   // Chip 1: (+U, +V)
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

   // Chip 2: (-U, +V)
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

   // Chip 3: (+U, -V)
   vcorner1 = center   + ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 + ChipSize * unit[0] - ChipSize * unit[1];
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

   // Chip 4: (-U, -V)
   vcorner1 = center   - ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 - ChipSize * unit[0] - ChipSize * unit[1];
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

// ========================================================================
//  Stage 1 Fit Functions (from MEGTXECPosLocalFit.cpp:443-534)
// ========================================================================

Double_t FitFuncU(Double_t* x, Double_t* par)
{
   // par: 0=scale(unused), 1=u, 2=w, 3=half_size (kSiPMSize/2 in PM units)
   if (par[2] == 0) par[2] = 0.0001;

   Double_t fitval =
      TMath::ATan2(((x[0] - par[1]) - par[3]), par[2]) -
      TMath::ATan2(((x[0] - par[1]) + par[3]), par[2]);

   Double_t norm =
      TMath::ATan2((0 - par[3]), par[2]) -
      TMath::ATan2((0 + par[3]), par[2]);

   if (norm != 0) {
      fitval /= norm;
   } else {
      fitval = 0;
   }
   return fitval;
}

Double_t FitFuncV(Double_t* x, Double_t* par)
{
   // par: 0=scale(unused), 1=v, 2=w, 3=half_phi, 4=radius
   if (par[3] == 0) par[3] = 0.0001;

   Double_t fitval =
      TMath::ATan2(par[4] * TMath::Sin((x[0] - par[1]) / par[4] + par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos((x[0] - par[1]) / par[4] + par[3]))) -
      TMath::ATan2(par[4] * TMath::Sin((x[0] - par[1]) / par[4] - par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos((x[0] - par[1]) / par[4] - par[3])));

   Double_t norm =
      TMath::ATan2(par[4] * TMath::Sin(0 + par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos(0 + par[3]))) -
      TMath::ATan2(par[4] * TMath::Sin(0 - par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos(0 - par[3])));

   if (norm != 0) {
      fitval /= norm;
   } else {
      fitval = 0;
   }
   return fitval;
}

Double_t FitFunc1PointU(Double_t* x, Double_t* par)
{
   // par: 0=scale, 1=u, 2=w, 3=half_size, 4=attenuation
   Double_t fitval = par[0] * FitFuncU(x, par)
                     * TMath::Exp(-TMath::Sqrt((x[0] - par[1]) * (x[0] - par[1]) + par[2] * par[2]) / par[4]);
   return fitval;
}

Double_t FitFunc1PointV(Double_t* x, Double_t* par)
{
   // par: 0=scale, 1=v, 2=w, 3=half_phi, 4=radius, 5=attenuation
   Double_t fitval = par[0] * FitFuncV(x, par)
                     * TMath::Exp(-TMath::Sqrt((x[0] - par[1]) * (x[0] - par[1]) + par[2] * par[2]) / par[5]);
   return fitval;
}

// ========================================================================
//  Stage 2: FCNSolid (TMinuit chi-square — modified to skip dead channels)
// ========================================================================

void FCNSolid(Int_t &npar, Double_t * /*gin*/, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
   // par: 0=Scale, 1=U, 2=V, 3=W
   gNDF = 0;
   TVector3 view = UVW2XYZVec(par[1], par[2], par[3]);
   Double_t chisq = 0;

   for (Int_t ch = 0; ch < kNInner; ch++) {
      if (!gPMUsed[ch]) continue;

      gNDF++;

      TVector3 pmPos(gPMPos[ch][0], gPMPos[ch][1], gPMPos[ch][2]);
      TVector3 pmNorm(gPMDir[ch][0], gPMDir[ch][1], gPMDir[ch][2]);
      Double_t solidangle = PMSolidAngleMPPC(view, pmPos, pmNorm);
      Double_t Nexp = par[0] * solidangle;

      // chi2 = (npho - Nexp)^2 / (npho / QE)
      // The denominator approximates the variance of the Poisson photoelectron count
      Double_t denom = gNphoArr[ch] / kQE_SiPM;
      if (denom < 1.0) denom = 1.0; // prevent division by zero
      chisq += (gNphoArr[ch] - Nexp) * (gNphoArr[ch] - Nexp) / denom;
   }

   gNDF -= npar;
   f = chisq;
}

// ========================================================================
//  Analytical Inner Face PM Geometry
// ========================================================================

void ComputeAnalyticalPMGeometry()
{
   // Compute positions and normals for inner-face SiPMs analytically.
   // Channel ch = row * kNCols + col, where col ∈ [0,43], row ∈ [0,92].
   // U (=z) goes along columns, V (=phi) goes along rows.

   for (Int_t ch = 0; ch < kNInner; ch++) {
      Int_t col = ch % kNCols;  // 0..43 → U direction
      Int_t row = ch / kNCols;  // 0..92 → V direction

      // UVW of this SiPM: position on inner face (W=0 by definition)
      Double_t u = (col - (kNCols - 1) / 2.0) * gPMIntervalU;
      Double_t v = -((row - (kNRows - 1) / 2.0) * gPMIntervalV); // negative sign: V convention
      Double_t w = 0.0; // on the inner face

      gPMU[ch] = u;
      gPMV[ch] = v;

      // Convert to XYZ
      Double_t uvw[3] = {u, v, w};
      Double_t xyz[3];
      MyUVW2XYZ(uvw, xyz);
      gPMPos[ch][0] = xyz[0];
      gPMPos[ch][1] = xyz[1];
      gPMPos[ch][2] = xyz[2];

      // Normal: for cylindrical inner face, outward radial direction (+x/r, +y/r, 0)
      Double_t r = TMath::Sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
      gPMDir[ch][0] = xyz[0] / r;
      gPMDir[ch][1] = xyz[1] / r;
      gPMDir[ch][2] = 0.0;
   }
}

// ========================================================================
//  Load dead channels from file
// ========================================================================

Int_t LoadDeadChannels(const char* deadFile)
{
   gDeadSet.clear();
   std::ifstream fin(deadFile);
   if (!fin.is_open()) {
      std::cerr << "ERROR: Cannot open dead channel file: " << deadFile << std::endl;
      return -1;
   }
   std::string line;
   while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') continue;
      Int_t ch = std::atoi(line.c_str());
      if (ch >= 0 && ch < kXECNChan) {
         gDeadSet.insert(ch);
      }
   }
   fin.close();
   return (Int_t)gDeadSet.size();
}

// ========================================================================
//  Main Macro
// ========================================================================

void LocalFitBaseline(const char* inputFile  = "",
                      const char* deadFile   = "",
                      const char* outPath    = "")
{
   if (strlen(inputFile) == 0) {
      std::cerr << "Usage: root -l -b -q 'LocalFitBaseline.C(\"MCGamma.root\", \"dead_channels.txt\" [, \"output.root\"])'\n";
      return;
   }

   // --- Load dead channels ---
   Int_t nDead = 0;
   if (strlen(deadFile) > 0) {
      nDead = LoadDeadChannels(deadFile);
      if (nDead < 0) return;
      std::cout << "Loaded " << nDead << " dead channels from " << deadFile << std::endl;
   } else {
      std::cout << "WARNING: No dead channel file specified. Running without dead channel exclusion.\n";
   }

   // --- Compute PM intervals ---
   // The official code computes fPMIntervalU from the run header PM positions.
   // For the standard MC geometry: 44 SiPMs span ~66.4 cm → spacing ≈ 1.5098 cm.
   // The projection fit works in "PM units" (bin index from -22 to +22).
   gPMIntervalU = 1.5097727; // cm — spacing between adjacent SiPMs in U direction
   gPMIntervalV = gXEPHI_rad * kReff / kNRows; // cm — spacing in V direction (~1.531 cm)

   std::cout << "PM intervals: U=" << gPMIntervalU << " cm, V=" << gPMIntervalV << " cm\n";

   // --- Compute analytical PM geometry ---
   ComputeAnalyticalPMGeometry();

   // Note: if run from within the MEG offline framework with MEGXECPMRunHeader available,
   // PM geometry from the run header could be loaded for higher accuracy. For standalone
   // use, the analytical geometry (computed above) is sufficient for the inner face.
   std::cout << "Using analytical PM geometry for inner face.\n";

   // --- Input file ---
   TFile* fIn = TFile::Open(inputFile, "READ");
   if (!fIn || fIn->IsZombie()) {
      std::cerr << "ERROR: Cannot open input file: " << inputFile << std::endl;
      return;
   }
   TTree* tree = (TTree*)fIn->Get("tree");
   if (!tree) {
      std::cerr << "ERROR: 'tree' not found in " << inputFile << std::endl;
      return;
   }

   // --- Input branches ---
   Int_t   run = 0, event = 0;
   Float_t npho[kXECNChan];
   Float_t uvwRecoFI[3], uvwTruth[3], xyzTruth[3];
   Float_t energyTruth = 0;

   tree->SetBranchAddress("run",        &run);
   tree->SetBranchAddress("event",      &event);
   tree->SetBranchAddress("npho",       npho);
   tree->SetBranchAddress("uvwRecoFI",  uvwRecoFI);
   tree->SetBranchAddress("uvwTruth",   uvwTruth);
   tree->SetBranchAddress("xyzTruth",   xyzTruth);
   tree->SetBranchAddress("energyTruth", &energyTruth);

   // --- Output file ---
   TString outName;
   if (strlen(outPath) > 0) {
      outName = outPath;
   } else {
      outName = inputFile;
      outName.ReplaceAll(".root", "_localfit.root");
   }
   TFile outFile(outName, "RECREATE", "Local Fit Baseline");

   // --- Output tree 1: predictions (one row per dead channel per event) ---
   Int_t    o_event_idx, o_sensor_id, o_face, o_mask_type;
   Long64_t o_run_number, o_event_number;
   Float_t  o_truth_npho, o_pred_npho, o_error_npho;

   TTree* predTree = new TTree("predictions", "Dead channel npho predictions");
   predTree->Branch("event_idx",    &o_event_idx,    "event_idx/I");
   predTree->Branch("run_number",   &o_run_number,   "run_number/L");
   predTree->Branch("event_number", &o_event_number, "event_number/L");
   predTree->Branch("sensor_id",    &o_sensor_id,    "sensor_id/I");
   predTree->Branch("face",         &o_face,         "face/I");
   predTree->Branch("mask_type",    &o_mask_type,    "mask_type/I");
   predTree->Branch("truth_npho",   &o_truth_npho,   "truth_npho/F");
   predTree->Branch("pred_npho",    &o_pred_npho,    "pred_npho/F");
   predTree->Branch("error_npho",   &o_error_npho,   "error_npho/F");

   // --- Output tree 2: position (one row per event) ---
   Int_t    p_event_idx;
   Long64_t p_run_number, p_event_number;
   Float_t  p_uvwTruth[3], p_uvwRecoFI[3], p_uvwFitNoDead[3], p_uvwStage1[3];
   Float_t  p_fitScale, p_fitChisq, p_energyTruth;
   Int_t    p_nPMUsed;

   TTree* posTree = new TTree("position", "Position reconstruction results");
   posTree->Branch("event_idx",     &p_event_idx,     "event_idx/I");
   posTree->Branch("run_number",    &p_run_number,    "run_number/L");
   posTree->Branch("event_number",  &p_event_number,  "event_number/L");
   posTree->Branch("uvwTruth",      p_uvwTruth,       "uvwTruth[3]/F");
   posTree->Branch("uvwRecoFI",     p_uvwRecoFI,      "uvwRecoFI[3]/F");
   posTree->Branch("uvwFitNoDead",  p_uvwFitNoDead,   "uvwFitNoDead[3]/F");
   posTree->Branch("uvwStage1",     p_uvwStage1,      "uvwStage1[3]/F");
   posTree->Branch("fitScale",      &p_fitScale,      "fitScale/F");
   posTree->Branch("fitChisq",      &p_fitChisq,      "fitChisq/F");
   posTree->Branch("nPMUsed",       &p_nPMUsed,       "nPMUsed/I");
   posTree->Branch("energyTruth",   &p_energyTruth,   "energyTruth/F");

   // --- Stage 1 fit functions ---
   // U fit: 5 params (scale, u, w, half_size, attenuation)
   TF1* fFitPointU = new TF1("fFitPointU", FitFunc1PointU, -kNCols / 2.0, kNCols / 2.0, 5);
   fFitPointU->FixParameter(3, kSiPMSize / (2.0 * gPMIntervalU)); // half-size in PM units
   fFitPointU->FixParameter(4, kAtten / gPMIntervalU);             // attenuation in PM units

   // V fit: 6 params (scale, v, w, half_phi, radius, attenuation)
   TF1* fFitPointV = new TF1("fFitPointV", FitFunc1PointV, -kNRows / 2.0, kNRows / 2.0, 6);
   fFitPointV->FixParameter(3, kPhiSiPM / 2.0);                // half phi of cathode
   fFitPointV->FixParameter(4, kXERIN / gPMIntervalV);         // radius in PM units
   fFitPointV->FixParameter(5, kAtten / gPMIntervalV);         // attenuation in PM units

   // --- Event loop ---
   Long64_t nEntries = tree->GetEntries();
   std::cout << "Processing " << nEntries << " events...\n";

   // Suppress ROOT info messages (e.g. "ParameterSettings: lower/upper bounds
   // outside current parameter value") during fitting loop
   Int_t savedErrorLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kWarning;

   Int_t nProcessed = 0, nFailed = 0;

   for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
      tree->GetEntry(iEntry);

      if (iEntry % 1000 == 0) {
         std::cout << "Event " << iEntry << "/" << nEntries << "\r" << std::flush;
      }

      // Skip events with invalid reconstruction
      if (!std::isfinite(uvwRecoFI[0]) || std::abs(uvwRecoFI[0]) > 1e5) continue;
      if (!std::isfinite(uvwTruth[0])  || std::abs(uvwTruth[0])  > 1e5) continue;

      // ================================================================
      //  Stage 1: Local Projection Fit (with dead channel exclusion)
      // ================================================================

      // --- Fill projection histograms, skipping dead channels ---
      TH1D hProjU("hProjU", "", kNCols, -kNCols / 2.0, kNCols / 2.0);
      TH1D hProjV("hProjV", "", kNRows, -kNRows / 2.0, kNRows / 2.0);
      Double_t binError2U[kNCols] = {0};
      Double_t binError2V[kNRows] = {0};

      for (Int_t ch = 0; ch < kNInner; ch++) {
         if (gDeadSet.count(ch)) continue; // SKIP dead channels

         Float_t n = npho[ch];
         if (!std::isfinite(n) || n <= 0 || n > 1e9) continue;

         Int_t col = ch % kNCols; // U bin
         Int_t row = ch / kNCols; // V bin

         hProjU.SetBinContent(col + 1, hProjU.GetBinContent(col + 1) + n);
         hProjV.SetBinContent(row + 1, hProjV.GetBinContent(row + 1) + n);

         // Error: sqrt(npho/QE) for Poisson stats
         Double_t uncert = TMath::Sqrt(n / kQE_SiPM);
         binError2U[col] += uncert * uncert;
         binError2V[row] += uncert * uncert;
      }

      for (Int_t i = 0; i < kNCols; i++)
         hProjU.SetBinError(i + 1, TMath::Sqrt(binError2U[i]));
      for (Int_t i = 0; i < kNRows; i++)
         hProjV.SetBinError(i + 1, TMath::Sqrt(binError2V[i]));

      // --- Fit U projection ---
      Double_t region = 10.0; // PM units (default)
      Double_t midu = uvwRecoFI[0] / gPMIntervalU;
      Double_t maxu = TMath::Min((Double_t)(kNCols / 2.0), midu + region);
      Double_t minu = TMath::Max((Double_t)(-kNCols / 2.0), midu - region);

      fFitPointU->SetParameter(0, hProjU.GetMaximum());
      fFitPointU->SetParameter(1, midu);
      fFitPointU->SetParameter(2, hProjU.GetRMS());
      fFitPointU->SetParLimits(1, midu - 0.5 * region, midu + 0.5 * region);
      Double_t rmsU = hProjU.GetRMS();
      if (rmsU < 0.5) rmsU = 1.0;
      fFitPointU->SetParLimits(2, 0.1 * rmsU, 1.5 * rmsU);

      hProjU.Fit("fFitPointU", "0QN", "", minu, maxu);
      Double_t stage1_u = fFitPointU->GetParameter(1) * gPMIntervalU;
      Double_t stage1_wu = fFitPointU->GetParameter(2) * gPMIntervalU;

      // --- Fit V projection ---
      Double_t midv = -uvwRecoFI[1] / gPMIntervalV; // note the negative sign
      Double_t maxv = TMath::Min((Double_t)(kNRows / 2.0), midv + region);
      Double_t minv = TMath::Max((Double_t)(-kNRows / 2.0), midv - region);

      fFitPointV->SetParameter(0, hProjV.GetMaximum());
      fFitPointV->SetParameter(1, midv);
      fFitPointV->SetParameter(2, hProjV.GetRMS());
      fFitPointV->SetParLimits(1, minv, maxv);
      Double_t rmsV = hProjV.GetRMS();
      if (rmsV < 0.5) rmsV = 1.0;
      fFitPointV->SetParLimits(2, 0.1 * rmsV, 1.5 * rmsV);

      hProjV.Fit("fFitPointV", "0QN", "", minv, maxv);
      Double_t stage1_v = -fFitPointV->GetParameter(1) * gPMIntervalV; // negative sign back
      Double_t stage1_wv = fFitPointV->GetParameter(2) * gPMIntervalV;

      // Combined Stage 1 result: U from U-fit, V from V-fit, W = average
      Double_t stage1_w = (stage1_wu + stage1_wv) / 2.0;
      if (stage1_w < 0.5) stage1_w = 0.5; // W must be positive (inside detector)

      p_uvwStage1[0] = stage1_u;
      p_uvwStage1[1] = stage1_v;
      p_uvwStage1[2] = stage1_w;

      // ================================================================
      //  Stage 2: Local Solid Angle Fit (with dead channel exclusion)
      // ================================================================

      // --- Select PMs for Stage 2: inner face, within region, npho>0, not dead ---
      Double_t regionCm_U = region * 1.05 * gPMIntervalU;
      Double_t regionCm_V = region * 1.05 * gPMIntervalV;
      Int_t nPMUsed = 0;
      Double_t nphoMax = -1;
      Double_t nphoSum = 0;

      for (Int_t ch = 0; ch < kNInner; ch++) {
         gPMUsed[ch] = kFALSE;
         gNphoArr[ch] = 0;

         if (gDeadSet.count(ch)) continue; // SKIP dead channels

         Float_t n = npho[ch];
         if (!std::isfinite(n) || n <= 0 || n > 1e9) continue;

         // Distance in UVW from Stage 1 position
         Double_t du = TMath::Abs(stage1_u - gPMU[ch]);
         Double_t dv = TMath::Abs(stage1_v - gPMV[ch]);

         if (du > regionCm_U || dv > regionCm_V) continue;

         gPMUsed[ch] = kTRUE;
         gNphoArr[ch] = n;
         nPMUsed++;
         nphoSum += n;
         if (n > nphoMax) nphoMax = n;
      }

      // Skip events with too few PMs
      if (nPMUsed < 10) {
         nFailed++;
         continue;
      }

      // --- MINUIT fit ---
      Double_t initialScale = nphoSum / 20.0; // rough initial scale

      static TMinuit* minuit = nullptr;
      Int_t ierflg;
      Double_t arglist[10];
      if (!minuit) {
         minuit = new TMinuit(4);
         minuit->SetFCN(FCNSolid);
         minuit->SetPrintLevel(-1);
         minuit->mnexcm("SET NOW", arglist, 0, ierflg);
         arglist[0] = 1;
         minuit->mnexcm("SET ERR", arglist, 1, ierflg);
         arglist[0] = 1;
         minuit->mnexcm("SET STR", arglist, 1, ierflg);
      }

      minuit->mnparm(0, "Scale", initialScale, 1e5, 0, 0, ierflg);
      minuit->mnparm(1, "U", stage1_u, 0.1, stage1_u - 5, stage1_u + 5, ierflg);
      minuit->mnparm(2, "V", stage1_v, 0.1, stage1_v - 5, stage1_v + 5, ierflg);
      minuit->mnparm(3, "W", stage1_w, 0.1, TMath::Max(0.1, stage1_w - 5), stage1_w + 5, ierflg);

      arglist[0] = 500;  // max iterations
      arglist[1] = 0.1;  // tolerance
      minuit->mnexcm("MIGRAD", arglist, 2, ierflg);

      // Get results
      Double_t fitU, fitV, fitW, fitScale;
      Double_t eU, eV, eW, eScale;
      minuit->GetParameter(0, fitScale, eScale);
      minuit->GetParameter(1, fitU, eU);
      minuit->GetParameter(2, fitV, eV);
      minuit->GetParameter(3, fitW, eW);

      Double_t edm, errdef, fcn;
      Int_t nvpar, nparx, icstat;
      minuit->mnstat(fcn, edm, errdef, nvpar, nparx, icstat);

      Double_t fitChisq = (gNDF > 0) ? fcn / gNDF : 1e6;

      // ================================================================
      //  Fill position output
      // ================================================================

      p_event_idx    = (Int_t)iEntry;
      p_run_number   = run;
      p_event_number = event;
      for (Int_t i = 0; i < 3; i++) {
         p_uvwTruth[i]  = uvwTruth[i];
         p_uvwRecoFI[i] = uvwRecoFI[i];
      }
      p_uvwFitNoDead[0] = fitU;
      p_uvwFitNoDead[1] = fitV;
      p_uvwFitNoDead[2] = fitW;
      p_fitScale      = fitScale;
      p_fitChisq      = fitChisq;
      p_nPMUsed       = nPMUsed;
      p_energyTruth   = energyTruth;
      posTree->Fill();

      // ================================================================
      //  Dead channel npho prediction
      // ================================================================

      TVector3 viewFit = UVW2XYZVec(fitU, fitV, fitW);

      for (Int_t ch : gDeadSet) {
         if (ch >= kNInner) continue; // only predict inner face

         o_event_idx    = (Int_t)iEntry;
         o_run_number   = run;
         o_event_number = event;
         o_sensor_id    = ch;
         o_face         = 0;        // inner face
         o_mask_type    = 0;        // MC has truth

         // Truth npho for this dead channel
         Float_t truthN = npho[ch];
         o_truth_npho = (std::isfinite(truthN) && truthN >= 0 && truthN < 1e9) ? truthN : 0;

         // Predicted npho = Scale * omega(fitted_pos, PM_ch)
         TVector3 pmPos(gPMPos[ch][0], gPMPos[ch][1], gPMPos[ch][2]);
         TVector3 pmNorm(gPMDir[ch][0], gPMDir[ch][1], gPMDir[ch][2]);
         Double_t omega = PMSolidAngleMPPC(viewFit, pmPos, pmNorm);
         o_pred_npho = fitScale * omega;

         o_error_npho = o_pred_npho - o_truth_npho;
         predTree->Fill();
      }

      nProcessed++;
   }

   gErrorIgnoreLevel = savedErrorLevel;

   std::cout << "\nProcessed " << nProcessed << " events, " << nFailed << " failed Stage 2.\n";

   // --- Write output ---
   outFile.cd();
   predTree->Write();
   posTree->Write();

   Long64_t nPred = predTree->GetEntries();
   Long64_t nPos  = posTree->GetEntries();

   outFile.Close();
   fIn->Close();

   std::cout << "Output written to: " << outName << std::endl;
   std::cout << "  predictions tree: " << nPred << " entries (dead ch predictions)\n";
   std::cout << "  position tree: "    << nPos  << " entries (event positions)\n";

   delete fFitPointU;
   delete fFitPointV;
}
