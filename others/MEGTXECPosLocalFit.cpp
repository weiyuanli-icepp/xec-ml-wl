// Author: Satoru Kobayashi, Shinji Ogawa

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// MEGTXECPosLocalFit                                                         //
//                                                                            //
// Begin_Html <!--
/*-->

Description:
<p>

Position reconstruction using solid angle and photon distribution 
locally.

For details of the algorithm and performance, please read <a 
href="https://savannah.psi.ch/repos/meg/websvn/log.php?repname=meg&path=%2
Ftrunk%2Fmegdocs%2Fnotes%2Fxec_analysis.pdf&rev=0&sc=0&isdir=0">megdocs/no
tes/xec_analysis.pdf</a>.

</p>

Usage:
<p>
This task requires that xecpmt.npho is filled in advance. So, at least, 
XECPMTFill task must be activated.
For DRS data, ReadData and XECWaveformAnalysis will calculate PMT charge 
to be filled at XECPMTFill task.

</p>

Status:
<p>
Usable for public.<br>
Currently fitting for (u,v) projection is the default.
Another "Method" using solid angle of each PMT is also available.
Since the new method has a better performance for shallow or very deep 
events, we are thinking to change the default to the new one.

</p>

To Do:
<p>

<ul>
  <li>Study of performance of "Method" with solid angle of each PMT by 
using experiment data.
  <li>Study to improve reconstruction for shallow events.
  <li>Improvement of document.
  <li>Estimation of resolution with taking into account effect of size of 
slits and decay vertex distribution.
</ul>

</p>

Known Problems:
<p>
With ROOT v5.22/00 and v5.22/00a warnings like
<pre>
MINUIT WARNING IN PARAMETR
 ============== VARIABLE2 BROUGHT BACK INSIDE LIMITS.
</pre>
are shown. This is due to a problem of ROOT.
This will be fixed at v5.22/00b.
<br>
<br>
Thre is a strong biass of reconstruction for shallower events than 2 cm. 
It tends to be reconstructed to a center of PMT cathode.

</p>

<!--*/
// --> End_Html
//                                                                            //
// The event methods have been written by Satoru Kobayashi and Shinji Ogawa.  //
//                                                                            //
// Please note: The following information is only correct after executing     //
// the ROMEBuilder.                                                           //
//                                                                            //
// This task accesses the following folders :                                 //
//     EventHeader                                                            //
//     XECRunHeader                                                           //
//     XECPMRunHeader                                                         //
//     XECPMCluster                                                           //
//     XECSumWaveform                                                         //
//     XECFastRecResult                                                       //
//     XECPosLocalFitResult                                                   //
//     XECPosLocalFitParameters                                               //
//     XECPosLocalFitCorrection                                               //
//     XECPileupResult                                                        //
//     XECPileupClusteringResult                                              //
//     XECClusterInfo                                                         //
//     XECClusterInfo2                                                        //
//     XECPoslMonitorPlots                                                    //
//     TRGXECOnline                                                           //
//     MCXECHit                                                               //
//                                                                            //
// This task contains the following histgrams :                               //
//    UVDist                                                                  //
//                                                                            //
// The histograms/graph are created and saved automaticaly by the task.       //
//                                                                            //
// The following method can be used to get a handle of the histogram/graph :  //
//                                                                            //
// Get<Histogram/Graph Name>()                                                //
//                                                                            //
// For histogram/graph arrays use :                                           //
//                                                                            //
// Get<Histogram/Graph Name>At(Int_t index)                                   //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

/* Generated header file containing necessary includes                        */
#include "generated/MEGTXECPosLocalFitGeneratedIncludes.h"

////////////////////////////////////////////////////////////////////////////////
/*  This header was generated by ROMEBuilder. Manual changes above the        *
 * following line will be lost next time ROMEBuilder is executed.             */
/////////////////////////////////////----///////////////////////////////////////
// $Id$

// temporarily disable warnings caused by various header file under Visual C++
#include <RConfig.h>
#if defined( R__VISUAL_CPLUSPLUS )
#pragma warning( push )
#pragma warning( disable : 4244 )
#pragma warning( disable : 4800 )
#endif // R__VISUAL_CPLUSPLUS
#include <TROOT.h>
#include <TMath.h>
#include <TF1.h>
#include <TF2.h>
#include <TVirtualFitter.h>
#include <TMinuit.h>
#include <TVector3.h>
#include <TStyle.h>
#include <TColor.h>
#include <TCanvas.h>
#include <TPostScript.h>
#include <TPaveText.h>
#include <TSpectrum2.h>
#include <TSpectrum.h>
#include <Math/Minimizer.h>
#if defined( R__VISUAL_CPLUSPLUS )
#pragma warning( pop )
#endif // R__VISUAL_CPLUSPLUS
#include "xec/PMSolidAngle.h"
#include "xec/xectools.h"
#include "constants/xec/xecconst.h"
#include "units/MEGPhysicalConstants.h"
#include "tasks/MEGTXEC.h"
#include "tasks/MEGTXECPosLocalFit.h"
#include "generated/MEGAnalyzer.h"
#include "generated/MEGXECClusterInfo.h"
#include "generated/MEGXECPoslMonitorPlots.h"
#include "ROMEiostream.h"
#include "ROMESQLDataBase.h"
#include "ROMEAnalyzer.h"

#include <Riostream.h>

using namespace std;
using namespace MEG;
using namespace XECTOOLS;
using namespace XECCONSTANTS;
using namespace PMSolidAngle;
namespace
{

// Parameters for local projection fit
const Double_t kSiPMSize = 1.2 * centimeter;        // Width of MPPC
const Double_t kPhiSiPM = TMath::ASin(1.2 / 64.8);  // Width of MPPC converted to phi
const Double_t kAtten = 1e6 * centimeter;           // Attenuation length

// Global variables
Int_t     gNDF(0);
Int_t     gGlobalNDF(0);
Double_t  gIncAngleThre;
Int_t gRegion;

// PMInfo class to store array of Npho etc.
class PMInfo
{
private:
   Double_t fNexp;
   Double_t fNpho;
   Double_t fNphoUncert;
   Double_t fNphe;
   TVector3 fXYZ;
   TVector3 fNorm;
   Int_t    fChannelNumber;
   Bool_t   fIsSiPM;
   Double_t fQE;
   Double_t fU;
   Double_t fV;
   Double_t fW;
   Double_t fUCluster;
   Double_t fVCluster;
   Double_t fWCluster;
   Short_t  fFace;
   Bool_t   fIsClustered;
   Int_t    fNPMClustered;
   Bool_t   fIsBad;
   Bool_t   fIsValid;
   Bool_t   fIsUsed4Outer;
   std::vector<Double_t> fNexpLSAF;
   std::vector<Bool_t>   fIsUsed4LSAF;
   Bool_t   fIsUsed4GSAF;
public:
   PMInfo(Int_t chnum);
   void SetChannelNumber(Int_t chnum)
   {
      fChannelNumber = chnum;
   };
   void SetU(Double_t u)
   {
      fU = u;
   };
   void SetV(Double_t v)
   {
      fV = v;
   };
   void SetW(Double_t w)
   {
      fW = w;
   };
   void SetUCluster(Double_t u)
   {
      fUCluster = u;
   };
   void SetVCluster(Double_t v)
   {
      fVCluster = v;
   };
   void SetWCluster(Double_t w)
   {
      fWCluster = w;
   };
   void SetXYZ(TVector3 XYZ)
   {
      fXYZ  = XYZ;
   };
   void SetXYZ(Double_t x, Double_t y, Double_t z)
   {
      fXYZ.SetXYZ(x, y, z);
   };
   void SetNorm(TVector3 Norm)
   {
      fNorm = Norm;
   };
   void SetNorm(Double_t x, Double_t y, Double_t z)
   {
      fNorm.SetXYZ(x, y, z);
   };
   void SetNpho(Double_t Npho)
   {
      fNpho = Npho;
   };
   void SetNphoUncert(Double_t NphoUncert)
   {
      fNphoUncert = NphoUncert;
   };
   void SetNphe(Double_t Nphe)
   {
      fNphe = Nphe;
   };
   void SetQE(Double_t QE)
   {
      fQE   = QE;
   };
   void SetFace(Short_t face)
   {
      fFace = face;
   };
   void SetNPMClustered(Int_t NPMClustered)
   {
      fNPMClustered = NPMClustered;
   };
   void SetIsUsed4LSAF(Int_t iregion, Bool_t isused)
   {
      fIsUsed4LSAF[iregion] = isused;
   };
   void SetIsUsed4GSAF(Bool_t isused)
   {
      fIsUsed4GSAF = isused;
   };
   void SetIsBad(Bool_t IsBad)
   {
      fIsBad = IsBad;
   };
   void SetIsValid(Bool_t IsValid)
   {
      fIsValid = IsValid;
   };
   void SetNexp(Double_t Nexp)
   {
      fNexp = Nexp;
   };
   void SetNexpLSAF(Int_t iregion, Double_t Nexp)
   {
      fNexpLSAF[iregion] = Nexp;
   };
   void InitNexpLSAF(Int_t nregion)
   {
      fNexpLSAF.resize(nregion);
   };
   void InitIsUsed4LSAF(Int_t nregion)
   {
      fIsUsed4LSAF.resize(nregion);
   };
   Bool_t   IsSiPM()
   {
      return fIsSiPM;
   };
   Int_t    GetChannelNumber()
   {
      return fChannelNumber;
   };
   Double_t GetNpho()
   {
      return fNpho;
   };
   Double_t GetNphoUncert()
   {
      return fNphoUncert;
   };
   Double_t GetNphe()
   {
      return fNphe;
   };
   Double_t GetU()
   {
      return fU;
   };
   Double_t GetV()
   {
      return fV;
   };
   Double_t GetW()
   {
      return fW;
   };
   Double_t GetUCluster()
   {
      return fUCluster;
   };
   Double_t GetVCluster()
   {
      return fVCluster;
   };
   Double_t GetWCluster()
   {
      return fWCluster;
   };
   Double_t GetDistanceOnUV(Double_t u, Double_t v);
   Double_t GetRelativeU(Double_t u);
   Double_t GetRelativeV(Double_t v);
   Short_t  GetFace()
   {
      return fFace;
   };
   TVector3 GetXYZ()
   {
      return fXYZ;
   };
   TVector3 GetNorm()
   {
      return fNorm;
   };
   Double_t GetQE()
   {
      return fQE;
   };
   Bool_t   GetIsBad()
   {
      return fIsBad;
   };
   Bool_t   GetIsValid()
   {
      return fIsValid;
   };
   Bool_t   IsClustered()
   {
      return fNPMClustered > 1 ? true : false;
   };
   Bool_t   IsUsed4LSAF(Int_t iregion)
   {
      return fIsUsed4LSAF[iregion];
   };
   Bool_t   IsUsed4GSAF()
   {
      return fIsUsed4GSAF;
   };
   Double_t GetNexpLSAF(Int_t iregion)
   {
      return fNexpLSAF[iregion];
   };
   Double_t GetNexp()
   {
      return fNexp;
   };
};

PMInfo::PMInfo(Int_t chnum)
{
   SetChannelNumber(chnum);
   if (chnum < 4092) {
      fIsSiPM = true;
   } else {
      fIsSiPM = false;
   }
}

Double_t PMInfo::GetRelativeU(Double_t u)
{
   return u - fU;
}

Double_t PMInfo::GetRelativeV(Double_t v)
{
   return v - fV;
}

Double_t PMInfo::GetDistanceOnUV(Double_t u, Double_t v)
{
   return TMath::Sqrt(TMath::Power(GetRelativeU(u), 2) + TMath::Power(GetRelativeV(v), 2));
}

std::map<Int_t, PMInfo*> PMArray;

// Function definition
TString safeName(TString name)
{
   TObject* old = gROOT->FindObject(name);
   if (old) {
      delete old;
   }
   return name;
}

// Fit function for local projection of Npho in U
Double_t FitFuncU(Double_t* x, Double_t* par)
{
   // nPar = 4
   // 0: scale(not used), 1: u, 2: w, 3:radius of cathode

   // Do not allow negative value of W
   if (par[2] == 0) {
      par[2] = 0.0001;
   }

   // 1D solid angle for PM at U = x[0]
   Double_t fitval =
      TMath::ATan2(((x[0] - par[1]) - par[3]), par[2]) -
      TMath::ATan2(((x[0] - par[1]) + par[3]), par[2]);

   // 1D solid angle for PM at U = par[1]
   Double_t norm =
      TMath::ATan2((0               - par[3]), par[2]) -
      TMath::ATan2((0               + par[3]), par[2]);

   // Relative 1D solid angle
   if (norm != 0) {
      fitval /= norm;
   } else {
      Report(R_WARNING, "Strange fit data or parameters. (x=%g,par=%g,%g,%g,%g)",
             x[0], par[0], par[1], par[2], par[3]);
      fitval = 0;
   }

   return fitval;
}

// Fit function for local projection of Npho in V
Double_t FitFuncV(Double_t* x, Double_t* par)
{
   // nPar = 5
   // 0: scale(not used), 1: v, 2: w, 3: half phi of cathode, 4: radius of inner face

   // Do not allow negative value of W
   if (par[3] == 0) {
      par[3] = 0.0001;
   }

   // 1D solid angle for PM at V = x[0]
   Double_t fitval =
      TMath::ATan2(par[4] * TMath::Sin((x[0] - par[1]) / par[4] + par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos((x[0] - par[1]) / par[4] + par[3]))) -
      TMath::ATan2(par[4] * TMath::Sin((x[0] - par[1]) / par[4] - par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos((x[0] - par[1]) / par[4] - par[3])));

   // 1D solid angle for PM at V = par[1]
   Double_t norm =
      TMath::ATan2(par[4] * TMath::Sin(0 + par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos(0 + par[3]))) -
      TMath::ATan2(par[4] * TMath::Sin(0 - par[3]),
                   (par[2] + par[4] - par[4] * TMath::Cos(0 - par[3])));

   // Relative 1D solid angle
   if (norm != 0) {
      fitval /= norm;
   } else {
      Report(R_WARNING, "Strange fit data or parameters. (x=%g,par=%g,%g,%g,%g,%g)",
             x[0], par[0], par[1], par[2], par[3], par[4]);
      fitval = 0;
   }

   return fitval;
}

// Fit function for local projection of Npho in U, including attenuation
Double_t FitFunc1PointU(Double_t* x, Double_t* par)
{
   // nPar = 5
   // 0: scale, 1: u, 2: w, 3: radius of cathode, 4: attenuation

   Double_t fitval = par[0] * FitFuncU(x, par)
                     * TMath::Exp(-1 * TMath::Sqrt((x[0] - par[1]) * (x[0] - par[1]) + par[2] * par[2]) / par[4]); // attenuation

   return fitval;
}

// Fit function for local projection of Npho in V, including attenuation
Double_t FitFunc1PointV(Double_t* x, Double_t* par)
{
   // nPar = 6
   // 0: scale, 1: v, 2: w, 3: half phi of cathode, 4: radius of inner face, 5: attenuation

   Double_t fitval = par[0] * FitFuncV(x, par)
                     * TMath::Exp(-1 * TMath::Sqrt((x[0] - par[1]) * (x[0] - par[1]) + par[2] * par[2]) / par[5]); // attenuation

   return fitval;
}

// Fit function for local solid angle fit
void FCNSolid(Int_t &npar, Double_t * /*gin*/, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
   // Fit U,V,W at same time using solid angle
   // nPar = 4
   // 0: scale, 1: u, 2: v, 3: w

   // Initialize variables
   gNDF = 0;
   TVector3 view(UVW2X(par[1], par[2], par[3]), UVW2Y(par[1], par[2], par[3]), UVW2Z(par[1], par[2], par[3]));
   Double_t chisq = 0, solidangle, anglecoeff, chisqipm, Nexp;

   // Calculate chi-square from the difference of Npho and Nexp
   for (Int_t iPM = 0; iPM < (Int_t)PMArray.size(); iPM++) {

      // Only use PM in specified region
      if (!PMArray[iPM]->IsUsed4LSAF(gRegion)) {
         continue;
      }
      gNDF++;

      // Calculate solid angle
      solidangle = PMSolidAngleMPPC(view, PMArray[iPM]->GetXYZ(), PMArray[iPM]->GetNorm());

      // Calculate Nexp
      anglecoeff = 1.;
      Nexp = par[0] * solidangle * anglecoeff;
      PMArray[iPM]->SetNexp(Nexp);

      // Calculate chi-square
      chisqipm = TMath::Power(PMArray[iPM]->GetNpho() - Nexp, 2) /
                 (PMArray[iPM]->GetNpho() / PMArray[iPM]->GetQE());
      chisq += chisqipm;

   }
   gNDF -= npar;
   f = chisq;
}

// Function for Global solid angle fit
void FCNGlobalSolid(Int_t &npar, Double_t * /*gin*/, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
   Double_t chisq = 0;
   gGlobalNDF = 0;
   TVector3 origin(
      UVW2X(par[0], par[1], par[2]),
      UVW2Y(par[0], par[1], par[2]),
      UVW2Z(par[0], par[1], par[2])
   );
   Double_t Theta = par[3] / radian;
   Double_t Phi  = par[4] / radian;
   TVector3 shwdir(
      TMath::Sin(Theta)*TMath::Cos(Phi),
      TMath::Sin(Theta)*TMath::Sin(Phi),
      TMath::Cos(Theta)
   );
   std::vector<Double_t> lenvec = {0, par[5]};
   std::vector<Double_t> scalevec = {par[6], par[7]};

   Int_t Nsource = lenvec.size();
   Double_t solidangle;
   Double_t incangle;
   Double_t anglecoeff;
   Double_t chisqipm;

   for (Int_t iPM = 0; iPM < (Int_t)PMArray.size(); iPM++) {
      if (!PMArray[iPM]->IsUsed4GSAF()) {
         continue;
      }
      Double_t Nexp = 0;
      for (Int_t isource = 0; isource < Nsource; isource++) {
         TVector3 source = origin + lenvec[isource] * shwdir;
         if (PMArray[iPM]->IsSiPM()) {
            solidangle = PMSolidAngleMPPC(source, PMArray[iPM]->GetXYZ(), PMArray[iPM]->GetNorm());
            incangle   = PMIncidentAngle(source, PMArray[iPM]->GetXYZ(), -PMArray[iPM]->GetNorm());
            anglecoeff = TransmittanceOfSilicon(incangle) / TransmittanceOfSilicon(0);
            Nexp       += scalevec[isource] * solidangle * anglecoeff;
         } else {
            solidangle = PMSolidAnglePMT(source, PMArray[iPM]->GetXYZ(), PMArray[iPM]->GetNorm());
            Nexp       += scalevec[isource] * solidangle;
         }
      }
      chisqipm = TMath::Power(PMArray[iPM]->GetNpho() - Nexp, 2) /
                 (PMArray[iPM]->GetNpho() / PMArray[iPM]->GetQE());

      chisq += chisqipm;
      gGlobalNDF++;
   }

   gGlobalNDF -= npar;
   f = chisq;
}

}

//______________________________________________________________________________
void MEGTXECPosLocalFit::Init()
{

   fFitPoint[0] = 0;
   fFitPoint[1] = 0;
   fProjectionU = 0;
   fProjectionV = 0;

   // Mnemonics for DRS/TRG loop.
   DoDRS = true;
   DoTRG = false;

   // Decide which fit to use
   MEGXECPosLocalFitParameters *pParameter = gAnalyzer->GetXECPosLocalFitParameters();
   if (GetSP()->GetMethod() >= 0) {
      pParameter->SetMethod(GetSP()->GetMethod());
   }
   SetSwitches(pParameter->GetMethod());
   fONNXTools = nullptr;
   if (fONNXRegressionSwitch) {
      InitializeONNXModel();
   }
   fScaleCorrectionSwitch = GetSP()->GetScaleCorrection();

   // Read trigger mask
   fTRGMask.clear();
   TObjArray *maskArray = GetSP()->GetTRGMask().Tokenize(",");
   if (!maskArray->GetEntries()) {
      maskArray = gAnalyzer->GetXECTask()->GetSP()->GetTRGMask().Tokenize(",");
   }
   for (Int_t iMask = 0; iMask < maskArray->GetEntries(); iMask++) {
      TObjString *mask = (TObjString*)maskArray->At(iMask);
      fTRGMask.push_back((Short_t)mask->GetString().Atoi());
   }
   delete maskArray;
}

void MEGTXECPosLocalFit::InitializeONNXModel()
{
   std::string fONNXModelPath = "../meg2posreg_20200927.onnx";

   fONNXTools = new MEGONNXTools((std::string)fONNXModelPath,
                                 gAnalyzer->GetGSP()->GetONNXGlobalSteering()->GetIntraOpNumThreads(),
                                 gAnalyzer->GetGSP()->GetONNXGlobalSteering()->GetInterOpNumThreads());

   size_t total_number_elements = fONNXTools->GetTotalNumberOfInputElements();
   if (total_number_elements == 0) {
      fONNXRegressionSwitch = false;
   }

   // std::cout<<"Initialized."<<std::endl;
}

void MEGTXECPosLocalFit::ONNXRegression(std::vector<Float_t> uvw)
{
   // std::cout<<"ONNX Regression"<<std::endl;
   auto &nphovector = fONNXTools->GetInputVector();
   nphovector.resize(4760);
   for (Int_t i = 0; i < 4760; i++) {
      nphovector[i] = (Float_t)PMArray[i]->GetNpho();
   }
   Float_t* output = fONNXTools->Predict()[0];
   // std::cout<<"output: "<<output[0]<<std::endl;
   // uvw[0] = output[];
   for (Int_t idim = 0; idim < 3; idim++) {
      uvw[idim] = output[idim];
   }
   // memcpy(uvw, &output[0], 3 * sizeof(Float_t));
}

void MEGTXECPosLocalFit::SetSwitches(Int_t FitMethod)
{
   // Decide which fit to use based on fit method variable
   // Initialize with all false
   fLocalProjectionFitSwitch  = kFALSE;
   fLocalSolidAngleFitSwitch  = kFALSE;
   fGlobalSolidAngleFitSwitch = kFALSE;
   fONNXRegressionSwitch      = kFALSE;

   switch (FitMethod) {
   case 0:
      fLocalProjectionFitSwitch  = kTRUE;
      fLocalSolidAngleFitSwitch = kFALSE;
      fGlobalSolidAngleFitSwitch = kFALSE;
      break;
   case 1:
   default:
      fLocalProjectionFitSwitch  = kTRUE;
      fLocalSolidAngleFitSwitch = kTRUE;
      fGlobalSolidAngleFitSwitch = kFALSE;
      break;
   case 2:
      fLocalProjectionFitSwitch  = kFALSE;
      fLocalSolidAngleFitSwitch = kFALSE;
      fGlobalSolidAngleFitSwitch = kFALSE;
      break;
   case 11:
      fLocalProjectionFitSwitch  = kTRUE;
      fLocalSolidAngleFitSwitch = kTRUE;
      fGlobalSolidAngleFitSwitch = kTRUE;
      break;
   case 1011:
      fLocalProjectionFitSwitch  = kTRUE;
      fLocalSolidAngleFitSwitch = kTRUE;
      fONNXRegressionSwitch = kTRUE;
   }
}

//______________________________________________________________________________
void MEGTXECPosLocalFit::BeginOfRun()
{

   // Get parameters used for corrections
   GetCorrectionParametersFromDB();
   if (fLocalSolidAngleFitSwitch) {
      GetCorrectionHistFromDB();
   }

   // Update parameters from SP
   MEGXECPosLocalFitParameters *pParameter = gAnalyzer->GetXECPosLocalFitParameters();
   if (GetSP()->GetCorrection()->GetSwitch() >= 0) {
      pParameter->SetSwitch(GetSP()->GetCorrection()->GetSwitch());
   }
   if (GetSP()->GetCorrection()->GetInterpolate() >= 0) {
      pParameter->SetInterpolate(GetSP()->GetCorrection()->GetInterpolate());
   }

   // Parameters to be used in fit etc.
   fActiveSizeU = gXEZIN;
   fActiveSizeV = gXEPHI / MEG::radian * gXERIN;
   fActiveSizeW = gXEROUT - gXERIN;
   fPMIntervalU = 1.5097727;
   fPMIntervalV = fActiveSizeV / kNUMPM2[kInnerFace];

   // Initialize local projection fit functions
   fFitPoint[0] = new TF1("fFitPointU", FitFunc1PointU, -kNUMPM1[0] / 2, kNUMPM1[0] / 2, 5);
   fFitPoint[0]->SetParNames("Scale", "Mean U", "Depth", "kSiPMSize/2", "Atten.");
   fFitPoint[0]->SetParameters(1, 0, 0, kSiPMSize / 2. / fPMIntervalU, kAtten);
   fFitPoint[0]->FixParameter(3, kSiPMSize / 2. / fPMIntervalU);
   fFitPoint[0]->FixParameter(4, kAtten);
   fFitPoint[1] = new TF1("fFitPointV", FitFunc1PointV, -kNUMPM2[0] / 2, kNUMPM2[0] / 2, 6);
   fFitPoint[1]->SetParNames("Scale", "Mean V", "Depth", "kPhiSiPM/2", "gXERIN", "Atten.");
   fFitPoint[1]->SetParameters(1, 0, 0, kPhiSiPM / 2. / fPMIntervalV, gXERIN, kAtten);
   fFitPoint[1]->FixParameter(3, kPhiSiPM / 2. / fPMIntervalV);
   fFitPoint[1]->FixParameter(4, gXERIN);
   fFitPoint[1]->FixParameter(5, kAtten);

   // Minimization method set to MIGRAD
   // If this is not called here,
   //  local projection fit for the first event is performed with the minimization method of TMinuit2/MIGRAD.
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit", "MIGRAD");
   //ROOT::Math::MinimizerOptions::PrintDefault();

   // Create local projection histograms to be fitted
   fProjectionU = new TH1F("fProjectionU", "U projection; U [MPPC]", kNUMPM1[0], -kNUMPM1[0] / 2.,
                           kNUMPM1[0] / 2.);
   fProjectionV = new TH1F("fProjectionV", "V projection; V [MPPC]", kNUMPM2[0], -kNUMPM2[0] / 2.,
                           kNUMPM2[0] / 2.);

   // Set position information in PM array
   Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   for (Int_t iPM = 0; iPM < nPM; iPM++) {
      PMInfo* upm = new PMInfo(iPM);
      upm->SetU(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetUVWAt(0));
      upm->SetV(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetUVWAt(1));
      upm->SetW(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetUVWAt(2));
      upm->SetXYZ(
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetXYZAt(0),
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetXYZAt(1),
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetXYZAt(2)
      );
      upm->SetNorm(
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetDirectionAt(0),
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetDirectionAt(1),
         gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetDirectionAt(2)
      );
      upm->SetFace(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetFace());
      upm->SetIsBad(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetIsBad());
      upm->SetQE(gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetQE());
      if (fLocalSolidAngleFitSwitch) {
         upm->InitNexpLSAF(fNregion);
         upm->InitIsUsed4LSAF(fNregion);
      }
      PMArray[iPM] = upm;
   }

   // Initialize plots to be shared with monitor
   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   poslPlots->Reset();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();
   TObjArray* func_posl = (TObjArray*)poslPlots->GetposlFuncs();
   for (Int_t iRegion = 0; iRegion < fNregion; iRegion++) {
      TF1* fProjFitU = new TF1(Form("fFitPointU_%d", iRegion), FitFunc1PointU, -kNUMPM1[0] / 2, kNUMPM1[0] / 2, 5);
      TF1* fProjFitV = new TF1(Form("fFitPointV_%d", iRegion), FitFunc1PointV, -kNUMPM2[0] / 2, kNUMPM2[0] / 2, 6);
      func_posl->Add(fProjFitU);
      func_posl->Add(fProjFitV);
      TH1F *fHProjU = new TH1F();
      fHProjU->SetName(Form("fHProjU_%d", iRegion));
      fHProjU->SetTitle(Form("U proj range%d; U [cm]", iRegion));
      TH1F *fHProjV = new TH1F();
      fHProjV->SetName(Form("fHProjV_%d", iRegion));
      fHProjV->SetTitle(Form("V proj range%d; V [cm]", iRegion));
      hist_posl->Add(fHProjU);
      hist_posl->Add(fHProjV);
      TH2F* fHNpho_exp = new TH2F(Form("fHNpho_exp%d", iRegion), Form("Region %d;column;row", iRegion), 44, 0, 44,
                                  93, 0, 93);
      hist_posl->Add(fHNpho_exp);
   }

   fHNpho = new TH2F(Form("fHNpho"), Form("Npho;column;row"), 44, 0, 44,
                     93, 0, 93);
   fHNpho_narrow = new TH2F(Form("fHNpho_narrow"), Form("Npho;column;row"), 28, 8, 36,
                            28, 44, 72);
   hist_posl->Add(fHNpho);

   fVerbose = GetSP()->GetVerbose();
}

void MEGTXECPosLocalFit::GetCorrectionParametersFromDB()
{
   // Get correction parameters related to regions of fitting
   MEGXECPosLocalFitParameters* pParameter = gAnalyzer->GetXECPosLocalFitParameters();
   fNregion = pParameter->GetNLocalRegion();
   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      fLocalregions.push_back(pParameter->GetLocalRegionsAt(iregion));
   }
   fRegionShift = pParameter->Getregionshift();
}

void MEGTXECPosLocalFit::GetCorrectionHistFromDB()
{
   // Get correction histograms for local solid angle fit from DB.
   // - Global Correction
   // - Shower Correction
   MEGXECPosLocalFitParameters* pParameter = gAnalyzer->GetXECPosLocalFitParameters();
   fGlobalCorrectionHistU.resize(fNregion);
   fGlobalCorrectionHistV.resize(fNregion);
   fGlobalCorrectionHistW.resize(fNregion);
   fShowerCorrectionHistU.resize(fNregion);
   fShowerCorrectionHistV.resize(fNregion);
   fShowerCorrectionHistW.resize(fNregion);

   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();

   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      MEGXECPosLocalFitCorrection* pCorrection = gAnalyzer->GetXECPosLocalFitCorrectionAt(iregion);
      fGlobalCorrectionHistU[iregion] = new TH1D(
         Form("hgc_%d_u", iregion),
         Form("Global range%d U", iregion),
         pParameter ->GetNBinUU(),
         pCorrection->GetXMinUU(),
         pCorrection->GetXMaxUU()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinUU(); ibin++) {
         fGlobalCorrectionHistU[iregion]->SetBinContent(ibin, pCorrection->GetBinUUAt(ibin));
      }

      fGlobalCorrectionHistV[iregion] = new TH1D(
         Form("hgc_%d_v", iregion),
         Form("Global range%d V", iregion),
         pParameter ->GetNBinVV(),
         pCorrection->GetXMinVV(),
         pCorrection->GetXMaxVV()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinVV(); ibin++) {
         fGlobalCorrectionHistV[iregion]->SetBinContent(ibin, pCorrection->GetBinVVAt(ibin));
      }

      fGlobalCorrectionHistW[iregion] = new TH1D(
         Form("hgc_%d_w", iregion),
         Form("Global range%d W", iregion),
         pParameter ->GetNBinWW(),
         pCorrection->GetXMinWW(),
         pCorrection->GetXMaxWW()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinWW(); ibin++) {
         fGlobalCorrectionHistW[iregion]->SetBinContent(ibin, pCorrection->GetBinWWAt(ibin));
      }

      fShowerCorrectionHistU[iregion] = new TH1D(
         Form("hsc_%d_u", iregion),
         Form("Shower range%d U", iregion),
         pParameter ->GetNBinSU(),
         pCorrection->GetXMinSU(),
         pCorrection->GetXMaxSU()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinSU(); ibin++) {
         fShowerCorrectionHistU[iregion]->SetBinContent(ibin, pCorrection->GetBinSUAt(ibin));
      }

      fShowerCorrectionHistV[iregion] = new TH1D(
         Form("hsc_%d_v", iregion),
         Form("Shower range%d V", iregion),
         pParameter ->GetNBinSV(),
         pCorrection->GetXMinSV(),
         pCorrection->GetXMaxSV()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinSV(); ibin++) {
         fShowerCorrectionHistV[iregion]->SetBinContent(ibin, pCorrection->GetBinSVAt(ibin));
      }

      fShowerCorrectionHistW[iregion] = new TH1D(
         Form("hsc_%d_w", iregion),
         Form("Shower range%d W", iregion),
         pParameter ->GetNBinSW(),
         pCorrection->GetXMinSW(),
         pCorrection->GetXMaxSW()
      );
      for (Int_t ibin = 0; ibin < pParameter ->GetNBinSW(); ibin++) {
         fShowerCorrectionHistW[iregion]->SetBinContent(ibin, pCorrection->GetBinSWAt(ibin));
      }

      // Add hists to ObjArray shared with monitor
      hist_posl->Add(fGlobalCorrectionHistU[iregion]);
      hist_posl->Add(fGlobalCorrectionHistV[iregion]);
      hist_posl->Add(fGlobalCorrectionHistW[iregion]);
      hist_posl->Add(fShowerCorrectionHistU[iregion]);
      hist_posl->Add(fShowerCorrectionHistV[iregion]);
      hist_posl->Add(fShowerCorrectionHistW[iregion]);

   }


}

//______________________________________________________________________________
void MEGTXECPosLocalFit::Event()
{
   if (!EventSelection()) {
      gAnalyzer->SetXECPosLocalFitResultSize(1);
      SetInvalidResult(1); // nGamma = 1
      return;
   }

   // Do DRS
   EventDigitizerIndependent(DoDRS);
}

// Main function to be called in each event
void MEGTXECPosLocalFit::EventDigitizerIndependent(Bool_t IsDRS)
{

   // Get nGamma
   Int_t nGamma;
   if (IsDRS) {
      if (gAnalyzer->GetXECPMClusterSize() > 0 && gAnalyzer->GetXECPMClusterAt(0)->Getfsize() > 1) {
         // [0]:as one gamma, [1]:main gamma. Not reconstruct pileup gamma.
         nGamma = 2;
      } else {
         nGamma = 1;
      }
   } else {
      nGamma = 0;
   }

   if (fVerbose) {
      std::cout << "nGamma: " << nGamma << std::endl;
   }

   gAnalyzer->SetXECPosLocalFitResultSize(nGamma);

   MEGXECPosLocalFitParameters* pParameter = gAnalyzer->GetXECPosLocalFitParameters();
   Double_t WLowerBound = pParameter->GetWLowerBound();

   // Prepare objects to draw in monitor
   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();
   // TObjArray* func_posl = (TObjArray*)poslPlots->GetposlFuncs();

   // Get nPM
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();

   // Loop for gamma
   for (Int_t iGamma = 0; iGamma < nGamma; iGamma++) {
      // Energy Scale
      Double_t EnergyScale = 1.;
      Int_t iGamma_cl = 0;
      if (fScaleCorrectionSwitch) {
         EnergyScale = gAnalyzer->GetXECFastRecResultAt(iGamma_cl)->Getenergy() / MEG::half_muon_mass_c2;
      }

      // Fill Nphe, Npho, UVW after clustering to array
      TH2F *fHNpho = (TH2F*)hist_posl->FindObject(Form("fHNpho"));
      fHNpho->Reset();

      for (Int_t iPM = 0; iPM < nPM; iPM++) {
         int iCluster = gAnalyzer->GetXECClusterInfo2At(iPM)->Getclusterid();
         int NPMClustered = gAnalyzer->GetXECPMClusterAt(iCluster)->Getclusterednpm();
         PMArray[iPM]->SetNPMClustered(NPMClustered);
         PMArray[iPM]->SetUCluster(gAnalyzer->GetXECPMClusterAt(iCluster)->GetUVWAt(0));
         PMArray[iPM]->SetVCluster(gAnalyzer->GetXECPMClusterAt(iCluster)->GetUVWAt(1));
         PMArray[iPM]->SetWCluster(gAnalyzer->GetXECPMClusterAt(iCluster)->GetUVWAt(2));
         PMArray[iPM]->SetIsValid(!gAnalyzer->GetXECPMClusterAt(iCluster)->GetinvalidAt(0));
         if (IsDRS) {
            Double_t Npho = TMath::Max(gAnalyzer->GetXECPMClusterAt(iCluster)->GetnphoAt(iGamma_cl) / NPMClustered,
                                       Float_t(0.));
            Double_t Nphe = TMath::Max(gAnalyzer->GetXECPMClusterAt(iCluster)->GetnpheAt(iGamma_cl) / NPMClustered,
                                       Float_t(0.));
            Double_t NphoUncert =  Nphe > 1. ? Npho / TMath::Sqrt(Nphe) : 1;
            PMArray[iPM]->SetNphe(Nphe);
            PMArray[iPM]->SetNpho(Npho);
            PMArray[iPM]->SetNphoUncert(NphoUncert);
         } else {
            PMArray[iPM]->SetNphe(0);
            PMArray[iPM]->SetNpho(0);
            PMArray[iPM]->SetNphoUncert(1);
         }
      }
      ReconstructBadChannels();
      for (Int_t iPM = 0; iPM < nPM; iPM++) {
         // fHNpho->SetBinContent(iPM % 44 + 1, iPM / 44 + 1, TMath::Max(0., PMArray[iPM]->GetNpho()));
         // std::cout<<"iPM: "<<iPM<<" IsBad: "<<PMArray[iPM]->GetIsBad()<<std::endl;
         fHNpho->SetBinContent(iPM % 44 + 1, iPM / 44 + 1, TMath::Max(0.1, PMArray[iPM]->GetNpho()));
         if (iPM % 44 >= 8 && iPM % 44 < 36 && iPM / 44 >= 44 && iPM / 44 < 72) {
            fHNpho_narrow->SetBinContent((iPM % 44) - 8 + 1, (iPM / 44) - 44 + 1, TMath::Max(0., PMArray[iPM]->GetNpho()));
         }
      }

      /* ----------------------------------------------------------------------------------
        First estimation is center position of max PM.
        PMT is weighted with 1/16.
        Consistency with trigger should be checked.
      ----------------------------------------------------------------------------------*/
      Double_t nphoMax     = 0;
      Double_t nphoMaxIn   = 0;
      Int_t    maxFace     = 0;
      Double_t maxUVW[3]   = {0., 0., 0.};
      Double_t inmaxUVW[3] = {0., 0., 0.};
      Int_t    maxPMIn     = 0;
      Double_t faceWeight[] = {1, 1. / 16, 1. / 16, 1. / 16, 1. / 16, 1. / 16};
      for (Int_t iPM = 0; iPM < nPM; iPM++) {

         // Find max PM in all faces
         if (PMArray[iPM]->GetNpho() / faceWeight[PMArray[iPM]->GetFace()] > nphoMax) {
            nphoMax = PMArray[iPM]->GetNpho() / faceWeight[PMArray[iPM]->GetFace()];
            maxUVW[0] = PMArray[iPM]->GetUCluster();
            maxUVW[1] = PMArray[iPM]->GetVCluster();
            maxUVW[2] = PMArray[iPM]->GetWCluster();
            maxFace = PMArray[iPM]->GetFace();
         }
      }

      // Find max PM in inner face
      maxPMIn = GetInMaxMPPC(GetSP()->GetInMaxMethod());
      nphoMaxIn = PMArray[maxPMIn]->GetNpho();
      inmaxUVW[0] = PMArray[maxPMIn]->GetUCluster();
      inmaxUVW[1] = PMArray[maxPMIn]->GetVCluster();
      inmaxUVW[2] = PMArray[maxPMIn]->GetWCluster();

      // Fill max PM result
      MEGXECPosLocalFitResult *Result;
      if (IsDRS) {
         Result = gAnalyzer->GetXECPosLocalFitResultAt(iGamma);
      } else {
         Result = 0;
      }
      Result->Setmaxface(maxFace);
      Result->Setmaxpmclustered(PMArray[maxPMIn]->IsClustered());
      Result->SetmaxuvwCopy(3, maxUVW);
      Result->SetinmaxuvwCopy(3, inmaxUVW);
      Result->Setinmaxpm(maxPMIn);
      poslPlots->SetmaxPM(maxPMIn);

      /* ----------------------------------------------------------------------------------
        Check the size of the region whose waveform is NOT cluster.
        Region center is set to inmaxUVW.
      ----------------------------------------------------------------------------------*/

      // Check non-cluster region around max PM
      Double_t NonClusterRegion_Square = fActiveSizeV * 1.1;
      Double_t NonClusterRegion_Circle = sqrt(fActiveSizeV * fActiveSizeV + fActiveSizeU * fActiveSizeU) * 1.1;
      Double_t NonClusterRegion_U = fActiveSizeU * 1.1;
      Double_t NonClusterRegion_V = fActiveSizeV * 1.1;
      for (Int_t iPM = 0; iPM < nPM; iPM++) {
         if (PMArray[iPM]->GetFace() != kInnerFace) {
            continue;
         }
         if (Result->Getmaxpmclustered()  == PMArray[iPM]->IsClustered()) {
            continue;
         }
         Double_t PMU = PMArray[iPM]->GetU();
         Double_t PMV = PMArray[iPM]->GetV();
         Double_t PMUCL = PMArray[iPM]->GetUCluster();
         Double_t PMVCL = PMArray[iPM]->GetVCluster();
         // Square
         if (TMath::Abs(PMU - inmaxUVW[0]) < NonClusterRegion_Square
             && TMath::Abs(PMV - inmaxUVW[1]) < NonClusterRegion_Square) {
            NonClusterRegion_Square = TMath::Max(TMath::Abs(PMU - inmaxUVW[0]), TMath::Abs(PMV - inmaxUVW[1]));
         }
         // Circle
         if (pow(PMU - inmaxUVW[0], 2.) + pow(PMV - inmaxUVW[1], 2.) < pow(NonClusterRegion_Circle, 2.)) {
            NonClusterRegion_Circle = sqrt(pow(PMU - inmaxUVW[0], 2.) + pow(PMV - inmaxUVW[1], 2.));
         }
         // U only
         if (TMath::Abs(PMU - inmaxUVW[0]) < NonClusterRegion_U
             && TMath::Abs(PMVCL - inmaxUVW[1]) < fPMIntervalV * 4 / 2.) {
            NonClusterRegion_U = TMath::Abs(PMU - inmaxUVW[0]);
         }
         // V only
         if (TMath::Abs(PMV - inmaxUVW[1]) < NonClusterRegion_V
             && TMath::Abs(PMUCL - inmaxUVW[0]) < fPMIntervalU * 4 / 2.) {
            NonClusterRegion_V = TMath::Abs(PMV - inmaxUVW[1]);
         }
      }

      // Fill noncluster region to Result
      Result->SetnonclusterregionAt(0, NonClusterRegion_Square);
      Result->SetnonclusterregionAt(1, NonClusterRegion_Circle);
      Result->SetnonclusterregionAt(2, NonClusterRegion_U);
      Result->SetnonclusterregionAt(3, NonClusterRegion_V);

      /* ----------------------------------------------------------------------------------
        Second estimation by local weighted mean
        Adjacement PMs are used. This range can be optimized in the future.
      ----------------------------------------------------------------------------------*/
      // Array to store final uvw.
      Double_t uvwFinal[3];
      Double_t uvwFinalUncert[3];

      // Set FitInitPM
      Int_t FitInitPM = maxPMIn;
      if (GetSP()->GetFitInitMethod() == 0) {
         // Use MC Truth for fit init
         if (gAnalyzer->GetMCXECHitSize() < 1) {
            continue;
         }
         for (int i = 0; i < gAnalyzer->GetMCXECHitSize(); i++) {
            if (gAnalyzer->GetMCXECHitAt(i)->GetIsMain()) {
               Double_t x_true = gAnalyzer->GetMCXECHitAt(i)->GetxyzAt(0);
               Double_t y_true = gAnalyzer->GetMCXECHitAt(i)->GetxyzAt(1);
               Double_t z_true = gAnalyzer->GetMCXECHitAt(i)->GetxyzAt(2);
               Double_t mc_uvw[3] = {XYZ2U(x_true, y_true, z_true),
                                     XYZ2V(x_true, y_true, z_true),
                                     XYZ2W(x_true, y_true, z_true)
                                    };
               FitInitPM = GetNearestMPPC(mc_uvw);
            }
         }
      }

      // Local weighted mean
      Double_t uvwmean[3] = {0., 0., 0.};
      LocalWeightedMean(FitInitPM, uvwmean);
      Result->SetuvwmeanCopy(3, uvwmean);
      memcpy(uvwFinal, &uvwmean[0], 3 * sizeof(Double_t));

      FillProjectedHistogram();

      /* ----------------------------------------------------------------------------------
        Fitting npho projected distribution
        You can select fixed fitting region or variable one depending on the depth by SP parameter.
        Fitting region information is hard corded, and there may be room for optimization.
      ----------------------------------------------------------------------------------*/
      // Initial uvw
      Double_t uvwInit[3];
      memcpy(uvwInit, &uvwmean[0], 3 * sizeof(Double_t));
      SetInitialUVW(uvwInit);

      std::vector<std::vector<Double_t>> uvwProFitUncorr(fNregion, std::vector<Double_t>(3));
      std::vector<std::vector<Double_t>> uvwProFitUncorrUncert(fNregion, std::vector<Double_t>(3));
      std::vector<Bool_t> LocalProjectionFitFlag(fNregion);
      if (fLocalProjectionFitSwitch) {
         // Buffer for fit result
         DoLocalProjectionFit(iGamma, uvwInit, uvwProFitUncorr, uvwProFitUncorrUncert, LocalProjectionFitFlag);
      } // End region loop

      // Global correction
      std::vector<std::vector<Double_t>> uvwProFitCorr(fNregion, std::vector<Double_t>(3));
      GlobalCorr_LocalProFit(uvwProFitUncorr, uvwProFitCorr, EnergyScale);

      Int_t iUsedRange = 0;
      for (Int_t iRegion = fNregion - 1; iRegion >= 0; --iRegion) {
         if (uvwProFitCorr[iRegion][2] > fLocalregions[iRegion] && // projection-fitted w > region
             LocalProjectionFitFlag[iRegion] && // should be always true as of July 2024
             Result->GetuvwprofitChisqAt(iRegion, 0) < GetSP()->GetLocalProjectionFit()->GetChisqThreshold() && // u projection fit chi2
             Result->GetuvwprofitChisqAt(iRegion, 1) < GetSP()->GetLocalProjectionFit()->GetChisqThreshold()) { // v projection fit chi2
            iUsedRange = iRegion;
            break;
         }
         //if (iRegion == 0) {
         //   Report(R_INFO, "Region used for projection fit is not specified. Use 0.");
         //}
      }

      Result->Setprofitrangeused(iUsedRange);

      // uvw ProjectionFit result after correction
      Double_t uvwProjectionFit[3] = {0., 0., 0.};
      Double_t uvwProjectionFitUncert[3] = {0., 0., 0.};
      for (Int_t idim = 0; idim < 3; idim++) {
         uvwProjectionFit[idim] = uvwProFitCorr[iUsedRange][idim];
         uvwProjectionFitUncert[idim] = uvwProFitUncorrUncert[iUsedRange][idim];
      }
      Result->SetuvwprofitCopy(3, uvwProjectionFit);
      Result->SetuvwprofitUncertCopy(3, uvwProjectionFitUncert);

      // Final uvw value at this point
      memcpy(uvwFinal, &uvwProjectionFit[0], 3 * sizeof(Double_t));
      memcpy(uvwFinalUncert, &uvwProjectionFitUncert[0], 3 * sizeof(Double_t));

      /* ----------------------------------------------------------------------------------
         Local Solid Analge Fit when specified in SP.
      ----------------------------------------------------------------------------------*/
      // Do not allow negative value of W
      for (Int_t iRegion = fNregion - 1; iRegion > 0; iRegion--) {
         if (uvwProFitCorr[iRegion][2] <= 0) {
            uvwProFitCorr[iRegion][2] = 0.001;
         }
      }

      Int_t iUsedRangeSolid = GetSP()->GetLocalSolidAngleFit()->GetFixedFitRegionIndex();
      if (iUsedRangeSolid < 0) {
         iUsedRangeSolid = GetSolidRegion(uvwProjectionFit[2]);
      }
      Double_t chisq_LSAF = 0;
      // Vectors to fill results
      std::vector<std::vector<Double_t>> uvwSolidRaw(fNregion, std::vector<Double_t>(3));
      std::vector<std::vector<Double_t>> uvwSolidRawUncert(fNregion, std::vector<Double_t>(3));

      if (fLocalSolidAngleFitSwitch) {
         // Loop for regions
         Bool_t invalidgamma = false;
         for (Int_t iRegion = 0; iRegion < fNregion; iRegion++) {
            if (!GetSP()->GetLocalSolidAngleFit()->GetFitAllRange()
                && iRegion != iUsedRangeSolid
                && iRegion != (iUsedRangeSolid + fRegionShift) % fNregion) {
               continue;
            }

            // Prepare variables for fit
            for (Int_t idim = 0; idim < 3; idim++) {
               uvwSolidRaw[iRegion][idim] = uvwProjectionFit[idim];
            }
            Float_t chisqSolid;
            Short_t fitResultSolid;

            // Local Solid Angle fit
            fitResultSolid = LocalSolidAngleFit(uvwSolidRaw[iRegion], uvwSolidRawUncert[iRegion], chisqSolid,  iRegion,
                                                true);
            if (fVerbose) {
               std::cout << iRegion
                         << " uvwRaw: " << uvwSolidRaw[iRegion][0] << " "
                         << uvwSolidRaw[iRegion][1] << " "
                         << uvwSolidRaw[iRegion][2] << std::endl;
            }

            // Set results
            Result->SetchisqSolidAt(iRegion, chisqSolid);
            Result->SetfitResultSolidAt(iRegion, fitResultSolid);
            for (Int_t idim = 0; idim < 3; idim++) {
               Result->SetuvwSolidbGCAt(iRegion, idim, uvwSolidRaw[iRegion][idim]);
               Result->SetuvwSolidbGCUncertAt(iRegion, idim, uvwSolidRawUncert[iRegion][idim]);
            }

            if (fitResultSolid < 0) {
               invalidgamma = true;
            }

            if (iRegion == iUsedRangeSolid) {
               chisq_LSAF = chisqSolid;
            }

            // Fill histogram to draw in monitor
            FillExpNphoHist(iRegion);

         } // End loop for region

         // Apply global correction
         std::vector<std::vector<Double_t>> uvwSolidAGC(fNregion, std::vector<Double_t>(3));
         std::vector<std::vector<Double_t>> uvwSolidAGCUncert(fNregion, std::vector<Double_t>(3));
         ApplyGlobalCorrection(uvwSolidRaw, uvwSolidRawUncert, uvwSolidAGC, uvwSolidAGCUncert, EnergyScale);

         // Apply shower correction
         std::vector<std::vector<Double_t>> uvwSolidASC(fNregion, std::vector<Double_t>(3));
         std::vector<std::vector<Double_t>> uvwSolidASCUncert(fNregion, std::vector<Double_t>(3));
         ApplyShowerCorrection(uvwSolidAGC, uvwSolidAGCUncert, uvwSolidASC, uvwSolidASCUncert, EnergyScale);

         // Fill results after corrections
         for (Int_t iRegion = 0; iRegion < fNregion; iRegion++) {
            for (Int_t idim = 0; idim < 3; idim++) {
               Result->SetuvwSolidbSCAt(iRegion, idim, uvwSolidAGC[iRegion][idim]);
               Result->SetuvwSolidbSCUncertAt(iRegion, idim, uvwSolidAGCUncert[iRegion][idim]);
               Result->SetuvwSolidAt(iRegion, idim, uvwSolidASC[iRegion][idim]);
               Result->SetuvwSolidUncertAt(iRegion, idim, uvwSolidASCUncert[iRegion][idim]);
            }
         }

         // Find good region to use
         if (invalidgamma) {
            iUsedRangeSolid = -1;
         }

         Result->SetLSfitrangeused(iUsedRangeSolid);

         // Fill results
         if (iUsedRangeSolid >= 0) {
            for (Int_t idim = 0; idim < 3; idim++) {
               uvwFinal[idim] = uvwSolidASC[iUsedRangeSolid][idim];
               uvwFinalUncert[idim] = uvwSolidASCUncert[iUsedRangeSolid][idim];
            }
            // Restrict w to positive with lower bound.
            switch (XECTOOLS::gXECUVWDEF) {
            case XECTOOLS::UVWDefinition::Global:
               uvwFinal[2] = TMath::Max(WLowerBound, uvwFinal[2]);
               break;
            case XECTOOLS::UVWDefinition::Local:
            case XECTOOLS::UVWDefinition::INVALID:
            default:
               uvwFinal[2] = TMath::Max(PMArray[GetNearestMPPC(uvwFinal)]->GetW() + WLowerBound,
                                        uvwFinal[2]);
               break;
            }
            if (fVerbose) {
               std::cout << "uvwRaw: " << uvwSolidRaw[iUsedRangeSolid][0] << " " << uvwSolidRaw[iUsedRangeSolid][1] << " " << uvwSolidRaw[iUsedRangeSolid][2] << std::endl;
               std::cout << "uvwFinal: " << uvwFinal[0] << " " << uvwFinal[1] << " " << uvwFinal[2] << std::endl;
            }
         }
      } // End if LocalSolidAngleFit switch

      /* ----------------------------------------------------------------------------------
        Run Global Solid Angle Fit if specified in switch
      ----------------------------------------------------------------------------------*/
      if (fGlobalSolidAngleFitSwitch) {
         Double_t ShowerInfo[3];
         Double_t scale[2];
         Double_t uvwGlobal[3];
         Double_t uvwGlobalUncert[3];
         memcpy(uvwGlobal, uvwProjectionFit, sizeof(uvwGlobal));
         Float_t chisqGlobal;
         GlobalSolidAngleFit(uvwGlobal, uvwGlobalUncert, ShowerInfo, scale, chisqGlobal);
         Result->SetuvwGlobalCopy(3, uvwGlobal);
         Result->Setshowertheta(ShowerInfo[0]);
         Result->Setshowerphi(ShowerInfo[1]);
         Result->Setshowerlength(ShowerInfo[2]);
         Result->SetGlobalFitscaleCopy(2, scale);
      } // End if GlobalSolidAngleFit switch
      // } // End if LocalProjectionFit switch

      std::vector<Float_t> uvwreg(3, 0);
      if (fONNXRegressionSwitch) {
         ONNXRegression(uvwreg);
         Result->SetuvwRegressionCopy(3, uvwreg.data());
      }

      /* ----------------------------------------------------------------------------------
        Fill folders.
      ----------------------------------------------------------------------------------*/

      // Final reconstructed position
      Result->SetuvwCopy(3, uvwFinal);
      Result->SetuvwUncertCopy(3, uvwFinalUncert);
      Result->Setchisq(chisq_LSAF);
      // const Int_t idxLightCenter = 3;
      if (iUsedRangeSolid >= 0) {
         for (Int_t idim = 0; idim < 3; idim++) {
            Result->SetuvwLightCenterAt(idim, uvwSolidRaw[iUsedRangeSolid][idim]);
         }
      } else {
         Result->SetuvwLightCenterAt(0, kXECInvalidU);
         Result->SetuvwLightCenterAt(1, kXECInvalidV);
         Result->SetuvwLightCenterAt(2, kXECInvalidW);
      }
      // std::cout<<"U: "<<Result->GetuvwLightCenterAt(0)<<" "<<Result->GetuvwLightCenterAt(1)<<std::endl;
      Result->SetxyzAt(0, UVW2X(uvwFinal[0], uvwFinal[1], uvwFinal[2]));
      Result->SetxyzAt(1, UVW2Y(uvwFinal[0], uvwFinal[1], uvwFinal[2]));
      Result->SetxyzAt(2, UVW2Z(uvwFinal[0], uvwFinal[1], uvwFinal[2]));

      // Max PM position
      Int_t nearpm = GetNearestMPPC(uvwFinal);
      Double_t LocalUV[2] = {PMArray[nearpm]->GetRelativeU(uvwFinal[0]), PMArray[nearpm]->GetRelativeV(uvwFinal[1])};
      Result->SetuvInMPPCCopy(2, LocalUV);
      Result->SetNearestMPPC(nearpm);

      // Depth
      Result->Setdepth(uvwFinal[2] - PMArray[nearpm]->GetW());

      // Used PM index list
      std::vector<Short_t> vUsedPM;
      for (Short_t iPM = 0; iPM < nPM; ++iPM) {
         if (PMArray[iPM]->IsUsed4LSAF(iUsedRangeSolid)) {
            vUsedPM.push_back(iPM);
         }
      }
      Result->SetusedPMIndexListSize(vUsedPM.size());
      Short_t idx(0);
      for (auto pm : vUsedPM) {
         Result->SetusedPMIndexListAt(idx++, pm);
      }

      // Solid angle from gamma to bad PMs
      Result->SetmaxSolidAngle2BadPM(GetMaximumSolidAngleToBadPM(Result->Getxyz()));
      Float_t sumSolidAngle(0.);
      TVector3 gammaxyz(Result->GetxyzAt(0), Result->GetxyzAt(1), Result->GetxyzAt(2));
      for (Short_t iPM = 0; iPM < nPM; ++iPM) {
         if (PMArray[iPM]->GetIsBad()) {
            if (PMArray[iPM]->IsSiPM()) {
               sumSolidAngle += PMSolidAngle::PMSolidAngleMPPC(gammaxyz, PMArray[iPM]->GetXYZ(), PMArray[iPM]->GetNorm()) * 4 * TMath::Pi();
            } else {
               sumSolidAngle += PMSolidAngle::PMSolidAnglePMT(gammaxyz, PMArray[iPM]->GetXYZ(), PMArray[iPM]->GetNorm()) * 4 * TMath::Pi();
            }
         }
      }
      Result->SetsummedSolidAngle2BadPM(sumSolidAngle);

      if (IsUVDistActive()) {
         GetUVDist()->Fill(uvwFinal[0], uvwFinal[1]);
      }

   } // End loop for gamma

}

//______________________________________________________________________________
void MEGTXECPosLocalFit::EndOfRun()
{
   // Delete members
   delete fProjectionU;
   delete fProjectionV;
   delete fFitPoint[0];
   delete fFitPoint[1];
   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   TObjArray* func_posl = (TObjArray*)poslPlots->GetposlFuncs();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();
   func_posl->Delete();
   hist_posl->Delete();
   delete fHNpho_narrow;
   for (auto &pm : PMArray) {
      delete pm.second;
   }
   PMArray.clear();
   SafeDelete(fONNXTools);
}

//______________________________________________________________________________
void MEGTXECPosLocalFit::Terminate()
{
}

//______________________________________________________________________________
Bool_t MEGTXECPosLocalFit::EventSelection()
{
   // Event selection method
   // 0: Based on nsum in FastRec
   // 1: Based on nsum in FastRec and # inner peaks in PLCL

   // Skip events with specific trigger masks
   if (fTRGMask.size()) {
      auto mask = gAnalyzer->GetEventHeader()->Getmask();
      if (std::find(fTRGMask.begin(), fTRGMask.end(), mask) != fTRGMask.end()) {
         return false;
      }
   }

   // Skip event if there is no FastRec result
   if (!gAnalyzer->GetXECFastRecResultSize()) {
      return false;
   }

   switch (GetSP()->GetEventSelectionMethod()) {
   case 0:
      // Skip event if xecfastrec.nsum < 1000
      if (gAnalyzer->GetXECFastRecResultAt(0)->Getninner() <= GetSP()->GetNinnerThreshold()) {
         return false;
      }
      break;
   case 1:
   default:
      // Skip event if xecfastrec.nsum < 1000 AND there is no inner LD peak.
      if (gAnalyzer->GetXECFastRecResultAt(0)->Getninner() > GetSP()->GetNinnerThreshold()) {
         return true;
      }
      if (!gAnalyzer->GetXECPileupClusteringResultSize()) {
         return false;
      } else {
         Short_t nInnerClusters(0);
         for (Short_t iGamma = 0; iGamma < gAnalyzer->GetXECPileupClusteringResultSize(); ++iGamma) {
            auto pPLCL = gAnalyzer->GetXECPileupClusteringResultAt(iGamma);
            if (pPLCL->Getmode() == 0) { // Inner LD search
               nInnerClusters++;
            }
         }
         if (!nInnerClusters) {
            return false;
         }
      }
      break;
   }

   return true;
}

//______________________________________________________________________________
void MEGTXECPosLocalFit::SetInvalidResult(Int_t nGamma)
{
   // Fill invalid results.

   std::vector<Double_t> vkInvalidUVW = {XECTOOLS::kXECInvalidU, XECTOOLS::kXECInvalidV, XECTOOLS::kXECInvalidW};
   std::vector<Double_t> vkInvalidXYZ = {XECTOOLS::kXECInvalidX, XECTOOLS::kXECInvalidY, XECTOOLS::kXECInvalidZ};

   for (Int_t iGamma = 0; iGamma < nGamma; iGamma++) {
      auto pResult = gAnalyzer->GetXECPosLocalFitResultAt(iGamma);

      for (Short_t idir = 0; idir < 3; ++idir) {
         pResult->SetxyzAt(idir, vkInvalidXYZ[idir]);
         pResult->SetxyzUncertAt(idir, 1e10);
         pResult->SetuvwAt(idir, vkInvalidUVW[idir]);
         pResult->SetuvwUncertAt(idir, 1e10);
      }
      pResult->Setchisq(1e10);
      pResult->Setdepth(XECTOOLS::kXECInvalidW);
      pResult->SetmaxSolidAngle2BadPM(-1000);
      pResult->SetsummedSolidAngle2BadPM(-1000);

      // Local solid angle method
      for (Short_t idir = 0; idir < 3; ++idir) {
         for (Short_t iRegion = 0; iRegion < fNregion; ++iRegion) {
            pResult->SetuvwSolidAt(iRegion, idir, vkInvalidUVW[idir]);
            pResult->SetuvwSolidUncertAt(iRegion, idir, 1e10);
            pResult->SetuvwSolidbGCAt(iRegion, idir, vkInvalidUVW[idir]);
            pResult->SetuvwSolidbGCUncertAt(iRegion, idir, 1e10);
            pResult->SetuvwSolidbSCAt(iRegion, idir, vkInvalidUVW[idir]);
            pResult->SetuvwSolidbSCUncertAt(iRegion, idir, 1e10);
         }
         pResult->SetuvwLightCenterAt(idir, vkInvalidUVW[idir]);
      }
      for (Short_t iRegion = 0; iRegion < fNregion; ++iRegion) {
         pResult->SetchisqSolidAt(iRegion, 1e10);
         pResult->SetfitResultSolidAt(iRegion, -1);
      }
      pResult->SetLSfitrangeused(-1);
      pResult->Setsolidangleinmax(-1);
      pResult->SetusedPMIndexListSize(0);

      // Global solid angle method
      for (Short_t idir = 0; idir < 3; ++idir) {
         pResult->SetuvwGlobalAt(idir, vkInvalidUVW[idir]);
      }
      pResult->Setshowertheta(1e10);
      pResult->Setshowerphi(1e10);
      pResult->Setshowerlength(-1);
      pResult->SetGlobalFitscaleAt(0, -1);
      pResult->SetGlobalFitscaleAt(1, -1);

      // Regression
      for (Short_t idir = 0; idir < 3; ++idir) {
         pResult->SetuvwRegressionAt(idir, vkInvalidUVW[idir]);
      }

      // Projection fit
      for (Short_t idir = 0; idir < 3; ++idir) {
         pResult->SetuvwprofitAt(idir, vkInvalidUVW[idir]);
         pResult->SetuvwprofitUncertAt(idir, 1e10);
         for (Short_t iRegion = 0; iRegion < fNregion; ++iRegion) {
            pResult->SetuvwprofitUncorrAt(iRegion, idir, vkInvalidUVW[idir]);
            pResult->SetuvwprofitUncorrUncertAt(iRegion, idir, 1e10);
         }
      }
      pResult->Setprofitrangeused(-1);
      for (Short_t iRegion = 0; iRegion < fNregion; ++iRegion) {
         pResult->SetprofitflagAt(iRegion, kFALSE);
         pResult->SetuvwprofitChisqAt(iRegion, 0, 1e10);
         pResult->SetuvwprofitChisqAt(iRegion, 1, 1e10);
      }

      // Nearest MPPC
      for (Short_t idir = 0; idir < 2; ++idir) {
         pResult->SetuvInMPPCAt(idir, vkInvalidUVW[idir]);
      }
      pResult->SetNearestMPPC(-1);

      // Misc
      for (Short_t idir = 0; idir < 3; ++idir) {
         pResult->SetuvwmeanAt(idir, vkInvalidUVW[idir]);
         pResult->SetmaxuvwAt(idir, vkInvalidUVW[idir]);
         pResult->SetinmaxuvwAt(idir, vkInvalidUVW[idir]);
      }
      pResult->Setmaxface(-1);
      pResult->Setmaxpmclustered(kFALSE);
      pResult->Setinmaxpm(-1);
      pResult->Setdistance2(-1);
      for (Short_t i = 0; i < 4; ++i) pResult->SetnonclusterregionAt(i, 0);
      pResult->Setedge(kFALSE);

   } // end of gamma loop
}

// Find nearest MPPC
Int_t MEGTXECPosLocalFit::GetNearestMPPC(Double_t* uvw)
{

   // Loop for inner PMs
   Double_t dist_min = 1e6;
   Int_t iPM_min = 0;
   for (auto it = PMArray.begin(); it != PMArray.end(); it++) {
      auto PM_this = it->second;
      if (PM_this->GetFace() != kInnerFace) {
         continue;
      }

      // Find PM at minimum distance
      Double_t dist = sqrt(pow(PM_this->GetU() - uvw[0], 2.) + pow(PM_this->GetV() - uvw[1], 2.));
      if (dist < dist_min) {
         dist_min = dist;
         iPM_min = PM_this->GetChannelNumber();
      }
   }

   return iPM_min;
}

Int_t MEGTXECPosLocalFit::GetInMaxMPPC(Int_t method)
{
   // Calculate max npho MPPC to be used as a seed of the PoslFit.
   // method 0: max npho MPPC
   // method 1: max npho MPPC in a max npho 4x4 Patch
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   int maxPMIn = 0;
   if (method == 0) {
      float nphoMaxIn = PMArray[0]->GetNpho();
      for (Int_t iPM = 0; iPM < nPM; iPM++) {
         if (PMArray[iPM]->GetFace() != kInnerFace) {
            continue;
         }
         if (PMArray[iPM]->GetNpho() > nphoMaxIn) {
            nphoMaxIn = PMArray[iPM]->GetNpho();
            maxPMIn = iPM;
         }
      }
   } else if (method == 1) {
      std::map<int, float> vNphoPatch; // patch id, npho
      for (Int_t iPM = 0; iPM < nPM; iPM++) {
         if (PMArray[iPM]->GetFace() != kInnerFace) {
            continue;
         }
         int iPatch = (iPM / kNUMPM1[kInnerFace] / 4) * (kNUMPM1[kInnerFace] / 4) + iPM % kNUMPM1[kInnerFace] / 4;
         vNphoPatch[iPatch] += PMArray[iPM]->GetNpho();
      }
      auto itr = std::max_element(vNphoPatch.begin(), vNphoPatch.end(), [](const pair<int, float>& p1, const pair<int, float>& p2) {
         return p1.second < p2.second;
      });
      int maxPatch = (*itr).first;

      std::map<int, float> vNpho; // channel, npho
      for (int i = 0; i < 4; i++) {
         for (int j = 0; j < 4; j++) {
            int ch = maxPatch % (kNUMPM1[kInnerFace] / 4) * 4 + maxPatch / (kNUMPM1[kInnerFace] / 4) * kNUMPM1[kInnerFace] * 4;
            ch += i * kNUMPM1[kInnerFace] + j;
            vNpho[ch] = PMArray[ch]->GetNpho();
         }
      }
      auto itr2 = std::max_element(vNpho.begin(), vNpho.end(), [](const pair<int, float>& p1, const pair<int, float>& p2) {
         return p1.second < p2.second;
      });
      maxPMIn = (*itr2).first;
   }
   return maxPMIn;
}

// Local weighted mean
void MEGTXECPosLocalFit::LocalWeightedMean(Int_t maxPM, Double_t* uvw)
{
   // Calculate the mean position by using the maximum output PM and its adjacement PMs.
   // If max PM at edge || adjacent is clsutered -> return the center of max PM.
   // If max PM itself is clustered,
   //   if it is edge cluster ->return center of cluster.
   //   else return weighted mean between adjacent clusters.

   uvw[0] = PMArray[maxPM]->GetUCluster();
   uvw[1] = PMArray[maxPM]->GetVCluster();
   uvw[2] = 0;
   Int_t step = (gAnalyzer->GetXECPosLocalFitResultAt(0)->Getmaxpmclustered()) ? 4 : 1;

   // U
   if (maxPM % kNUMPM1[0] - step >= 0 && maxPM % kNUMPM1[0] + step < kNUMPM1[0]) {
      // not edge of detector.
      Int_t AdjPMUS = maxPM - step;
      Int_t AdjPMDS = maxPM + step;
      if (PMArray[AdjPMUS]->GetIsValid() && PMArray[AdjPMDS]->GetIsValid()) {
         // not edge of readout area.
         if (PMArray[AdjPMUS]->IsClustered() == PMArray[maxPM]->IsClustered()
             && PMArray[AdjPMDS]->IsClustered() == PMArray[maxPM]->IsClustered()) {
            // not edge of non-clustered (or clustered) area.
            uvw[0] = (
                        PMArray[AdjPMUS]->GetUCluster() * PMArray[AdjPMUS]->GetNpho() +
                        PMArray[maxPM]->GetUCluster()   * PMArray[maxPM]  ->GetNpho() +
                        PMArray[AdjPMDS]->GetUCluster() * PMArray[AdjPMDS]->GetNpho()
                     ) / (PMArray[AdjPMUS]->GetNpho() + PMArray[maxPM]  ->GetNpho() + PMArray[AdjPMDS]->GetNpho());
         }
      }
   }

   // V
   if (maxPM - step * kNUMPM1[0] >= 0 && maxPM + step * kNUMPM1[0] < kNUMPM1[0] * kNUMPM2[0]) {
      // not edge of detector.
      Int_t AdjPMTop = maxPM - step * kNUMPM1[0];
      Int_t AdjPMBot = maxPM + step * kNUMPM1[0];
      if (PMArray[AdjPMTop]->GetIsValid() && PMArray[AdjPMBot]->GetIsValid()) {
         // not edge of readout area.
         if (PMArray[AdjPMTop]->IsClustered() == PMArray[maxPM]->IsClustered()
             && PMArray[AdjPMBot]->IsClustered() == PMArray[maxPM]->IsClustered()) {
            // not edge of non-clustered (or clustered) area.
            uvw[1] = (
                        PMArray[AdjPMTop]->GetVCluster() * PMArray[AdjPMTop]->GetNpho() +
                        PMArray[maxPM]->GetVCluster()   * PMArray[maxPM]  ->GetNpho() +
                        PMArray[AdjPMBot]->GetVCluster() * PMArray[AdjPMBot]->GetNpho()
                     ) / (PMArray[AdjPMTop]->GetNpho() + PMArray[maxPM]  ->GetNpho() + PMArray[AdjPMBot]->GetNpho());
         }
      }
   }
   return;
}

void  MEGTXECPosLocalFit::DoLocalProjectionFit(Int_t  iGamma,
                                               Double_t* uvwInit,
                                               std::vector<std::vector<Double_t>> &uvwProFitUncorr,
                                               std::vector<std::vector<Double_t>> &uvwProFitUncorrUncert,
                                               std::vector<Bool_t> &LocalProjectionFitFlag)
{
   MEGXECPosLocalFitResult *Result;
   if (!gAnalyzer->GetXECPosLocalFitResultSize()) {
      return;
   } else {
      Result = gAnalyzer->GetXECPosLocalFitResultAt(iGamma);
   }

   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();
   TObjArray* func_posl = (TObjArray*)poslPlots->GetposlFuncs();

   // Fit in each region
   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      // Local projection fit
      Double_t uvwUProFit[3] = {0., 0., 0.};
      Double_t uvwVProFit[3] = {0., 0., 0.};
      Double_t uvwUProFitUncert[3] = {0., 0., 0.};
      Double_t uvwVProFitUncert[3] = {0., 0., 0.};
      Double_t uvwProFitChisq[2] = {0., 0.};
      LocalProjectionFitFlag[iregion] = LocalProjectionFit(uvwInit, uvwUProFit, uvwVProFit,
                                                           uvwUProFitUncert, uvwVProFitUncert, uvwProFitChisq, fLocalregions[iregion], Result->GetnonclusterregionAt(0));

      // Fill to vector
      uvwProFitUncorr[iregion][0] = uvwUProFit[0];
      uvwProFitUncorr[iregion][1] = uvwVProFit[1];
      uvwProFitUncorrUncert[iregion][0] = uvwUProFitUncert[0];
      uvwProFitUncorrUncert[iregion][1] = uvwVProFitUncert[1];

      // Fow W, temporary use fit result of U Projection
      uvwProFitUncorr[iregion][2] = uvwUProFit[2];
      uvwProFitUncorrUncert[iregion][2] = uvwUProFitUncert[2];

      // Fill result
      Result->SetprofitflagAt(iregion, LocalProjectionFitFlag[iregion]);
      for (Int_t idim = 0; idim < 3; idim++) {
         Result->SetuvwprofitUncorrAt(iregion, idim, uvwProFitUncorr[iregion][idim]);
         Result->SetuvwprofitUncorrUncertAt(iregion, idim, uvwProFitUncorrUncert[iregion][idim]);
      }
      Result->SetuvwprofitChisqAt(iregion, 0, uvwProFitChisq[0]);
      Result->SetuvwprofitChisqAt(iregion, 1, uvwProFitChisq[1]);

      // Copy histogram for monitor
      TH1F *hU = (TH1F*)hist_posl->FindObject(Form("fHProjU_%d", iregion));
      TH1F *hV = (TH1F*)hist_posl->FindObject(Form("fHProjV_%d", iregion));
      *hU = *fProjectionU;
      *hV = *fProjectionV;
      hU->SetName(Form("fHProjU_%d", iregion));
      hV->SetName(Form("fHProjV_%d", iregion));

      // Copy functions for monitor
      TF1* fitU = (TF1*)func_posl->FindObject(Form("fFitPointU_%d", iregion));
      TF1* fitV = (TF1*)func_posl->FindObject(Form("fFitPointV_%d", iregion));
      for (Int_t iPar = 0; iPar < fitU->GetNpar(); iPar++) {
         Float_t val_this = fFitPoint[0]->GetParameter(iPar);
         fitU->SetParameter(iPar, val_this);
      }
      for (Int_t iPar = 0; iPar < fitV->GetNpar(); iPar++) {
         Float_t val_this = fFitPoint[1]->GetParameter(iPar);
         fitV->SetParameter(iPar, val_this);
      }
   }
}

void MEGTXECPosLocalFit::FillProjectedHistogram()
{
   // Fill projection histogram
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   const bool kMaxPMClustered = gAnalyzer->GetXECPosLocalFitResultAt(0)->Getmaxpmclustered();

   // Initializations. Reset rebin.
   fProjectionU->Reset();
   fProjectionV->Reset();
   fProjectionU->SetBins(kNUMPM1[0], -kNUMPM1[0] / 2., kNUMPM1[0] / 2.);
   if (!kMaxPMClustered) {
      fProjectionV->SetBins(kNUMPM2[0], -kNUMPM2[0] / 2., kNUMPM2[0] / 2.);
   } else {
      fProjectionV->SetBins(kNUMPM2[0] - 1, -kNUMPM2[0] / 2., kNUMPM2[0] / 2. - 1);
   }

   // Fill projected histograms
   Double_t binError2U[100] = {0};
   Double_t binError2V[100] = {0};
   for (Int_t iPM = 0; iPM < nPM; iPM++) {
      if (PMArray[iPM]->GetFace() != kInnerFace) {
         continue;
      }
      if (kMaxPMClustered && iPM / kNUMPM1[0] == kNUMPM2[0] - 1) {
         continue;
      }
      // Fill histograms. Only use PMs with Npho > 0
      if (PMArray[iPM]->GetNpho() > 0 && PMArray[iPM]->GetNphoUncert() > 0) {
         Int_t Ubin = PMArray[iPM]->GetChannelNumber() % kNUMPM1[0];
         Int_t Vbin = PMArray[iPM]->GetChannelNumber() / kNUMPM1[0];
         fProjectionU->SetBinContent(Ubin + 1, fProjectionU->GetBinContent(Ubin + 1) + PMArray[iPM]->GetNpho());
         fProjectionV->SetBinContent(Vbin + 1, fProjectionV->GetBinContent(Vbin + 1) + PMArray[iPM]->GetNpho());
         binError2U[Ubin] += TMath::Power(PMArray[iPM]->GetNphoUncert(), 2);
         binError2V[Vbin] += TMath::Power(PMArray[iPM]->GetNphoUncert(), 2);
      }
      // }
   }

   // Set errors in histograms
   for (Int_t iBin = 0; iBin < fProjectionU->GetNbinsX(); iBin++) {
      fProjectionU->SetBinError(iBin + 1, TMath::Sqrt(binError2U[iBin]));
   }
   for (Int_t iBin = 0; iBin < fProjectionV->GetNbinsX(); iBin++) {
      fProjectionV->SetBinError(iBin + 1, TMath::Sqrt(binError2V[iBin]));
   }

   // If all PM clustered, fit histogram of less granularity.
   if (kMaxPMClustered) {
      fProjectionU->Rebin(4);
      fProjectionV->Rebin(4);
   }
}

// Local projection fit
Bool_t MEGTXECPosLocalFit::LocalProjectionFit(Double_t* uvwinit, Double_t* uvwUProFit, Double_t* uvwVProFit,
                                              Double_t*uvwUProFitUncert, Double_t* uvwVProFitUncert,  Double_t* uvwProFitChisq, Double_t region, Double_t NonClusterRegion_Square)
{
   // Reconstruct the position of gamma-ray by fitting photon distribution in one dimension
   // The 1d light distribution in u and v direction of the inner face is fitted by simple function.
   // Model: All photons comes from a single point and propagate without scattering and absorption
   // Fit range is specified by SP->LocalRegion in PM unit

// If nonclusterregion is samller than fitting region, return.
   if (NonClusterRegion_Square < TMath::Min(region * fPMIntervalU, region * fPMIntervalV) / 1.05) {
      memcpy(uvwUProFit, uvwinit, sizeof(uvwinit) * 3);
      memcpy(uvwVProFit, uvwinit, sizeof(uvwinit) * 3);
      return kFALSE;
   }

   // Initialize parameters of U fit function
   Double_t midu = uvwinit[0] / fPMIntervalU;
   Double_t maxu = TMath::Min((Double_t)(+kNUMPM1[0] / 2.), midu + region);
   Double_t minu = TMath::Max((Double_t)(-kNUMPM1[0] / 2.), midu - region);
   fFitPoint[0]->SetParameter(0, fProjectionU->GetMaximum());
   // fFitPoint[0]->SetParameter(1, fProjectionU->GetMean());
   fFitPoint[0]->SetParameter(1, uvwinit[0] / fPMIntervalU);
   fFitPoint[0]->SetParameter(2, fProjectionU->GetRMS());
   fFitPoint[0]->SetParError(0, 0);
   fFitPoint[0]->SetParError(1, 0);
   fFitPoint[0]->SetParError(2, 0);
   // fFitPoint[0]->SetParLimits(1, minu, maxu);
   fFitPoint[0]->SetParLimits(1, midu - 0.5 * region, midu + 0.5 * region);
   fFitPoint[0]->SetParLimits(2, 0.1 * fProjectionU->GetRMS(), 1.5 * fProjectionU->GetRMS());
   // fFitPoint[0]->SetParLimits(2, 0, 100);

   // Fit U
   fProjectionU->Fit("fFitPointU", "0Q", "", minu, maxu); // Fit option B? Fit only once?
   if (fProjectionU->GetFunction("fFitPointU") && region > 1) {
      uvwProFitChisq[0] = fProjectionU->GetFunction("fFitPointU")->GetChisquare() / (region * 2. + 1 - 3);
   }
   uvwUProFit[0] = fFitPoint[0]->GetParameter(1) * fPMIntervalU;
   uvwUProFit[2] = fFitPoint[0]->GetParameter(2) * fPMIntervalU;
   uvwUProFitUncert[0] = fFitPoint[0]->GetParError(1) * fPMIntervalU;
   uvwUProFitUncert[2] = fFitPoint[0]->GetParError(2) * fPMIntervalU;

   // Initialize parameters of V fit function
   Double_t maxv = TMath::Min((Double_t)(+kNUMPM2[0] / 2.), -uvwinit[1] / fPMIntervalV + region);
   Double_t minv = TMath::Max((Double_t)(-kNUMPM2[0] / 2.), -uvwinit[1] / fPMIntervalV - region);
   fFitPoint[1]->SetParameter(0, fProjectionV->GetMaximum());
   // fFitPoint[1]->SetParameter(1, fProjectionV->GetMean());
   fFitPoint[1]->SetParameter(1, -uvwinit[1] / fPMIntervalV);
   fFitPoint[1]->SetParameter(2, fProjectionV->GetRMS());
   fFitPoint[1]->SetParError(0, 0);
   fFitPoint[1]->SetParError(1, 0);
   fFitPoint[1]->SetParError(2, 0);
   fFitPoint[1]->SetParLimits(1, minv, maxv);
   // fFitPoint[1]->SetParLimits(2, 0, 100);
   fFitPoint[1]->SetParLimits(2, 0.1 * fProjectionV->GetRMS(), 1.5 * fProjectionV->GetRMS());

   // Fit V
   fProjectionV->Fit("fFitPointV", "0Q", "", minv, maxv);
   if (fProjectionV->GetFunction("fFitPointV") && region > 1) {
      uvwProFitChisq[1] = fProjectionV->GetFunction("fFitPointV")->GetChisquare() / (region * 2. + 1 - 3);
   }
   uvwVProFit[1] = -fFitPoint[1]->GetParameter(1) * fPMIntervalV;
   uvwVProFit[2] =  fFitPoint[1]->GetParameter(2) * fPMIntervalV;
   uvwVProFitUncert[1] = fFitPoint[1]->GetParError(1) * fPMIntervalV;
   uvwVProFitUncert[2] = fFitPoint[1]->GetParError(2) * fPMIntervalV;

   return kTRUE;
}

// Local solid angle fit
Int_t MEGTXECPosLocalFit::LocalSolidAngleFit(std::vector<Double_t> &uvw, std::vector<Double_t> &euvw,
                                             Float_t& chisq, Int_t iregion, Bool_t circleRegion)
{
   // Reconstruct the position of gamma-ray by fitting photon distribution
   // The light distribution of the inner face is fitted by simple model.
   // Model: All photons comes from a single point and propagate without scattering and absorption
   // Chi square function for MINUIT minimizer is specified above as FCNSolid.
   // Fit range is specified by SP->LocalRegion in PM unit

   // Turn off segmentation violation
   gAnalyzer->GetApplication()->DisableFPETrap();

   // Initialize parameters
   gIncAngleThre = 90 * degree;
   Double_t region = fLocalregions[iregion];
   Double_t maxdist2 = region * region * 1.05 * 1.05;
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   gRegion = iregion;

   // Set PMarray flags and find maxPM
   Int_t maxPM = 0;
   Double_t nphoMax = -1;
   Int_t NPMused = 0;
   for (Int_t iPM = 0; iPM < nPM; iPM++) {

      // Select PMs of interest
      PMArray[iPM]->SetIsUsed4LSAF(iregion, false);
      if (PMArray[iPM]->GetFace() != kInnerFace || PMArray[iPM]->GetNpho() <= 0) {
         continue;
      }

      // Skip if PM is inside region
      Double_t udist = TMath::Abs(uvw[0] - PMArray[iPM]->GetU());
      Double_t vdist = TMath::Abs(uvw[1] - PMArray[iPM]->GetV());
      if (circleRegion) {
         Double_t dist2 = TMath::Power(udist / fPMIntervalU, 2) + TMath::Power(vdist / fPMIntervalV, 2);
         if (dist2 > maxdist2) {
            continue;
         }
      } else if (udist > region * 1.05 * fPMIntervalU || vdist > region * 1.05 * fPMIntervalV) {
         continue;
      }

      // Set flag
      PMArray[iPM]->SetIsUsed4LSAF(iregion, true);
      NPMused++;

      // Find max Npho PM
      if (PMArray[iPM]->GetNpho() > nphoMax) {
         nphoMax = PMArray[iPM]->GetNpho();
         maxPM = iPM;
      }
   }

   // Return if there are no PMs in region
   if (nphoMax == -1 || NPMused < region * region / 2) {
      chisq = 1e6;
      for (Int_t idim = 0; idim < 3; idim++) {
         uvw[idim]  = 1e6;
         euvw[idim] = 1e6;
      }
      return -1;
   }

   // w should be large enough to calculate solid angle.
   if (uvw[2] < 1) {
      uvw[2] = 1;
   }

   // Get max solid angle
   TVector3 xyz(UVW2X(uvw[0], uvw[1], uvw[2]), UVW2Y(uvw[0], uvw[1], uvw[2]), UVW2Z(uvw[0], uvw[1], uvw[2]));

   // Set initial scale
   // Double_t maxSolidAngle = PMSolidAngleMPPC(xyz, PMArray[maxPM]->GetXYZ(), PMArray[maxPM]->GetNorm());
   // Double_t initialScale = nphoMax / maxSolidAngle;
   Double_t initialScale = gAnalyzer->GetXECFastRecResultAt(0)->Getnsum2() / 20;

   // Initialize MINUIT
   static TMinuit *minuit2 = 0;
   Double_t arglist[10];
   Int_t ierflg;
   if (!minuit2) {
      minuit2 = new TMinuit(4);                       // Initialize TMinuit with 4 parameters
      minuit2->SetFCN(FCNSolid);                      // Set chisquare function
      if (!fVerbose) {
         minuit2->SetPrintLevel(-1);                     // Printlevel=-1 means quiet
      }

      minuit2->mnexcm("SET NOW", arglist, 0, ierflg); // Supress warnings
      arglist[0] = 1;
      minuit2->mnexcm("SET ERR", arglist, 1, ierflg); // Error=1 means normal chisquare fit
      arglist[0] = 1;
      minuit2->mnexcm("SET STR", arglist, 1, ierflg); // Strategy=1 means normal number of function calls
   }

   // Set initial values and step size for parameters
   minuit2->mnparm(0, "Scale", initialScale, 1e5, 0, 0, ierflg);
   minuit2->mnparm(1, "U", uvw[0], 1e-1, uvw[0] - 5, uvw[0] + 5, ierflg);
   minuit2->mnparm(2, "V", uvw[1], 1e-1, uvw[1] - 5, uvw[1] + 5, ierflg);
   minuit2->mnparm(3, "W", uvw[2], 1e-1, uvw[2] - 5, uvw[2] + 5, ierflg);

   // Run minimization
   arglist[0] = 500;
   arglist[1] = 0.1;
   minuit2->mnexcm("MIGRAD", arglist, 2, ierflg);

   // Get minimization status
   Double_t edm, errdef, fcn;
   Int_t    nvpar, nparx, icstat;
   minuit2->mnstat(fcn, edm, errdef, nvpar, nparx, icstat);

   // Calculate chi-square
   if (gNDF > 0) {
      chisq = fcn / gNDF;
   }

   // Get Parameters
   minuit2->GetParameter(1, uvw[0], euvw[0]);
   minuit2->GetParameter(2, uvw[1], euvw[1]);
   minuit2->GetParameter(3, uvw[2], euvw[2]);

   // gAnalyzer->GetApplication()->EnableFPETrap();

   // Set Nexp to PMArray
   for (Int_t iPM = 0; iPM < nPM; iPM++) {
      if (!PMArray[iPM]->IsUsed4LSAF(iregion)) {
         continue;
      }
      PMArray[iPM]->SetNexpLSAF(iregion, PMArray[iPM]->GetNexp());
   }

   return ierflg;
   delete minuit2;

}

void MEGTXECPosLocalFit::GlobalCorr_LocalProFit(std::vector<std::vector<Double_t>> &uvwUncorr,
                                                std::vector<std::vector<Double_t>> &uvwCorr,
                                                Double_t energyscale = 1.)
{
   // Apply global correction for local projection fit results
   MEGXECPosLocalFitCorrection* pCorrection = 0;
   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      pCorrection = gAnalyzer->GetXECPosLocalFitCorrectionAt(iregion);
      uvwCorr[iregion][0] = uvwUncorr[iregion][0] - energyscale * uvwUncorr[iregion][0] * pCorrection->GetUCorr();
      uvwCorr[iregion][1] = uvwUncorr[iregion][1] - energyscale * uvwUncorr[iregion][1] * pCorrection->GetVCorr();
      uvwCorr[iregion][2] = uvwUncorr[iregion][2]
                            - energyscale * (pCorrection->GetWCorrAt(0)
                                             + pCorrection->GetWCorrAt(1) * uvwUncorr[iregion][2]
                                             + pCorrection->GetWCorrAt(2) * uvwUncorr[iregion][2] * uvwUncorr[iregion][2]);
   }
   return;
}

// Global solid angle fit
void MEGTXECPosLocalFit::GlobalSolidAngleFit(Double_t* uvw, Double_t* euvw, Double_t* ShowerInfo,
                                             Double_t* scale, Float_t& chisq)
{
   // for MINUIT
   static TMinuit *minuit = 0;
   Double_t arglist[10];
   Int_t    ierflg = 0;
   Double_t edm, errdef, fcn;
   Int_t    nvpar, nparx, icstat;

   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   Int_t       iPM;

   gIncAngleThre = 90 * degree;

   TVector3 xyz(
      UVW2X(uvw[0], uvw[1], uvw[2]),
      UVW2Y(uvw[0], uvw[1], uvw[2]),
      UVW2Z(uvw[0], uvw[1], uvw[2])
   );

   Double_t Initheta = xyz.Theta() * radian;
   Double_t Iniphi  = xyz.Phi()  * radian;
   if (Initheta < 0) {
      Initheta = Initheta + 360 * degree;
   }
   if (Iniphi < 0) {
      Iniphi   = Iniphi   + 360 * degree;
   }

   Int_t maxPM(0);
   Double_t  nphoMax = -1;
   for (iPM = 0; iPM < nPM; iPM++) {
      PMArray[iPM]->SetIsUsed4GSAF(false);
      if (PMArray[iPM]->GetNpho() <= 50) {
         continue;
      }
      if (PMArray[iPM]->GetIsBad()) {
         continue;
      }
      // if (InterruptedByInner(xyz,PMArray[iPM]->GetXYZ(),PMArray[iPM]->GetNorm())) {PMArray[iPM]->SetIsUsed4GSAF(false);continue;}
      Double_t incangle = (xyz - PMArray[iPM]->GetXYZ()).Angle(PMArray[iPM]->GetNorm()) * radian;
      if (incangle > gIncAngleThre) {
         continue;
      }
      // if (PMArray[iPM]->GetFace()!=kOuterFace) {PMArray[iPM]->SetIsUsed4GSAF(false);continue;}
      // if (!PMArray[iPM]->IsSiPM())                           {PMArray[iPM]->SetIsUsed4GSAF(false);continue;}

      // uncert2Used.push_back(npho[iPM] / qe);
      if (PMArray[iPM]->GetNpho() > nphoMax) {
         nphoMax = PMArray[iPM]->GetNpho();
         maxPM = iPM;
      }
      PMArray[iPM]->SetIsUsed4GSAF(true);
   }

   Double_t maxSolidAngle = PMSolidAngleMPPC(xyz, PMArray[maxPM]->GetXYZ(), PMArray[maxPM]->GetNorm());
   Double_t initialScale = 0;
   if (maxSolidAngle > 0) {
      initialScale = nphoMax / maxSolidAngle;
   } else {
      std::cout << "max solid angle is too small: " << maxSolidAngle << std::endl;
   }


   // Set up fitter "MINUIT"
   TVirtualFitter::SetDefaultFitter("Minuit");
   if (!minuit) {
      // minuit = new TMinuit(4); // initialize TMinuit with 5 parameters
      minuit = new TMinuit(8); // initialize TMinuit with 5 parameters
      // Set FCN
      minuit->SetFCN(FCNGlobalSolid); // set chisquare function

      minuit->SetPrintLevel(-1);
      minuit->mnexcm("SET NOW", arglist, 0, ierflg);
      arglist[0] = 1;
      minuit->mnexcm("SET ERR", arglist, 1, ierflg); // UP = 1 : normal chisquare fit
      arglist[0] = 1;
      minuit->mnexcm("SET STR", arglist, 1, ierflg); // normal
   }

   // Set initial values and step size for parameters

   minuit->mnparm(0, "U", uvw[0], 1e-4, -40, 40, ierflg);
   minuit->mnparm(1, "V", uvw[1], 1e-4, -80, 80, ierflg);
   minuit->mnparm(2, "W", uvw[2], 1e-4, 0, 40, ierflg);
   minuit->mnparm(3, "Scale1", initialScale, 1, 1e4, 10e8, ierflg);
   minuit->mnparm(3, "Theta", Initheta, 1e-4, Initheta - 90 * degree, Initheta + 90 * degree, ierflg);
   minuit->mnparm(4, "Phi", Iniphi, 1e-4, Iniphi  - 90 * degree, Iniphi  + 90 * degree, ierflg);
   minuit->mnparm(5, "Length", 5, 1e-4, 1, 30, ierflg);
   minuit->mnparm(6, "Scale1", initialScale, 1, 1e4, 10e8, ierflg);
   minuit->mnparm(7, "Scale2", initialScale, 1, 1e4, 10e8, ierflg);
   // Fix parameters
   // minuit->FixParameter(0);
   // minuit->FixParameter(1);
   // minuit->FixParameter(2);

   arglist[0] = 500;
   arglist[1] = 0.1;
   minuit->mnexcm("MIGRAD", arglist, 2, ierflg); // execute MIGRAD minimization

   // Print results
   minuit->mnstat(fcn, edm, errdef, nvpar, nparx, icstat);
   //minuit->mnprin(3, fcn);

   if (gGlobalNDF > 0) {
      chisq = fcn / gGlobalNDF;
   }

   Double_t ShowerThetaUncert;
   Double_t ShowerPhiUncert;
   Double_t ShowerLengthUncert;
   Double_t escale[2];
   // Get Parameters
   minuit->GetParameter(0, uvw[0], euvw[0]);
   minuit->GetParameter(1, uvw[1], euvw[1]);
   minuit->GetParameter(2, uvw[2], euvw[2]);
   minuit->GetParameter(3, ShowerInfo[0], ShowerThetaUncert);
   minuit->GetParameter(4, ShowerInfo[1], ShowerPhiUncert);
   minuit->GetParameter(5, ShowerInfo[2], ShowerLengthUncert);
   minuit->GetParameter(6, scale[0], escale[0]);
   minuit->GetParameter(7, scale[1], escale[1]);
   delete minuit;
}

void MEGTXECPosLocalFit::ApplyGlobalCorrection(std::vector<std::vector<Double_t>> &RawUVW,
                                               std::vector<std::vector<Double_t>> &RawUVWUncert,
                                               std::vector<std::vector<Double_t>> &AGCUVW,
                                               std::vector<std::vector<Double_t>> &AGCUVWUncert,
                                               Double_t energyscale)
{
   // Function to apply global correction after LS fit
   Double_t CorrectionU, CorrectionV, CorrectionW;
   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      CorrectionU = energyscale * GetHistCorrection(1, RawUVW[iregion][0], fGlobalCorrectionHistU[iregion]);
      CorrectionV = energyscale * GetHistCorrection(1, RawUVW[iregion][1], fGlobalCorrectionHistV[iregion]);
      CorrectionW = energyscale * GetHistCorrection(1, RawUVW[iregion][2], fGlobalCorrectionHistW[iregion]);
      AGCUVW[iregion][0] = RawUVW[iregion][0] - CorrectionU;
      AGCUVW[iregion][1] = RawUVW[iregion][1] - CorrectionV;
      AGCUVW[iregion][2] = RawUVW[iregion][2] - CorrectionW;
      AGCUVWUncert[iregion][0] = RawUVWUncert[iregion][0];
      AGCUVWUncert[iregion][1] = RawUVWUncert[iregion][1];
      AGCUVWUncert[iregion][2] = RawUVWUncert[iregion][2];
   }
}

void MEGTXECPosLocalFit::ApplyShowerCorrection(std::vector<std::vector<Double_t>> &AGCUVW,
                                               std::vector<std::vector<Double_t>> &AGCUVWUncert,
                                               std::vector<std::vector<Double_t>> &ASCUVW,
                                               std::vector<std::vector<Double_t>> &ASCUVWUncert,
                                               Double_t energyscale)
{
   // Function to apply shower correction after LS fit
   Double_t DistanceU, DistanceV, DistanceW;
   Double_t CorrectionU, CorrectionV, CorrectionW;
   for (Int_t iregion = 0; iregion < fNregion; iregion++) {
      DistanceU   = AGCUVW[iregion][0] - AGCUVW[(iregion + fRegionShift) % fNregion][0];
      DistanceV   = AGCUVW[iregion][1] - AGCUVW[(iregion + fRegionShift) % fNregion][1];
      DistanceW   = AGCUVW[iregion][2] - AGCUVW[(iregion + fRegionShift) % fNregion][2];
      CorrectionU = energyscale * GetHistCorrection(1, DistanceU, fShowerCorrectionHistU[iregion]);
      CorrectionV = energyscale * GetHistCorrection(1, DistanceV, fShowerCorrectionHistV[iregion]);
      CorrectionW = energyscale * GetHistCorrection(1, DistanceW, fShowerCorrectionHistW[iregion]);
      ASCUVW[iregion][0] = AGCUVW[iregion][0] - CorrectionU;
      ASCUVW[iregion][1] = AGCUVW[iregion][1] - CorrectionV;
      ASCUVW[iregion][2] = AGCUVW[iregion][2] - CorrectionW;
      ASCUVWUncert[iregion][0] = AGCUVWUncert[iregion][0];
      ASCUVWUncert[iregion][1] = AGCUVWUncert[iregion][1];
      ASCUVWUncert[iregion][2] = AGCUVWUncert[iregion][2];
   }
}

Int_t MEGTXECPosLocalFit::PM2Cluster(Int_t iPM)
{
   // Find cluster ID from iPM
   Int_t nCluster = gAnalyzer->GetXECClusterInfoSize();
   if (nCluster <= 0) {
      return -1;
   }

   // Loop for clusters
   for (Int_t iCluster = 0; iCluster < nCluster; iCluster++) {

      // Loop for PMs in cluster
      Int_t nPMinCluster = gAnalyzer->GetXECClusterInfoAt(iCluster)->Getnpmincluster();
      for (Int_t iPMinCluster = 0; iPMinCluster < nPMinCluster; iPMinCluster++) {
         Int_t iPM_this = gAnalyzer->GetXECClusterInfoAt(iCluster)->GetchannellistAt(iPMinCluster);
         if (iPM_this == iPM) {
            return iCluster;
         }
      }

   }
   return -1;
}

Int_t MEGTXECPosLocalFit::GetSolidRegion(const Double_t uvwprofit_w)
{
   // Get the best fit range to use in Local Solid Angle Fit.
   Int_t rtnregion = -1;
   for (Int_t i = fNregion - 2; i > 0; i--) {
      if (uvwprofit_w > fLocalregions[i]) {
         rtnregion = i;
         break;
      }
   }
   if (rtnregion < 0) {
      rtnregion = 0;
   }

   return rtnregion;

}

void MEGTXECPosLocalFit::ReconstructBadChannels()
{
   // Get nPM
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   // Reconstruct npho of bad channel
   for (Int_t iPM = 0; iPM < nPM; iPM++) {
      if (!PMArray[iPM]->GetIsValid() && !PMArray[iPM]->GetIsBad()) {
         continue;
      }
      if (PMArray[iPM]->GetNpho() < 1e-09 || PMArray[iPM]->GetIsBad()) {
         Double_t avg = 0;
         Double_t avgerr = 0;
         if (iPM % 44 > 0 && iPM % 44 < 43 && iPM / 44 > 0 && iPM / 44 < 92) {
            std::vector<Int_t> surroundPM = {iPM - 44, iPM - 1, iPM + 1, iPM + 44};

            Int_t nvalid = 0;
            for (Int_t isurpm = 0; isurpm < 4; isurpm++) {
               Int_t surpm = surroundPM[isurpm];
               if (PMArray[surpm]->GetNpho() > 1e-09) {
                  nvalid++;
                  avg += PMArray[surpm]->GetNpho();
                  avgerr += PMArray[surpm]->GetNphoUncert();
               }
            }
            if (nvalid > 0) {
               avg /= nvalid;
               avgerr /= nvalid;
            }

         }
         PMArray[iPM]->SetNpho(avg);
         PMArray[iPM]->SetNphoUncert(avgerr);
      }
   }
}

//______________________________________________________________________________
void MEGTXECPosLocalFit::SetInitialUVW(Double_t* uvwInit)
{
   // Get initial uv(w) position depending on SP, FitInitMethod.
   // -1:              Default configuration, Ones digit.
   // 0:               Local weighted UV position based on MC truth.
   // Ones digit:      Local weighted UV position based on PM with max Npho.
   // Tens digit:      XECPL results.
   //                  If multi-peak found and trigger patch disabled, peak with the largest energy selected.
   // Hundreds digit:  Trigger patch. This can be active with XECPL results switch.
   //                  Peak closest to triggered patch selected.
   // Thousands digit: XECPLCL results.
   //                  If multi-peak found, select cluster based on its time and Npho.
   //                  cluster time - PMT SumWF cftime within threshold (30 ns) and largest Npho.

   Short_t iMainGamma(-1), nGamma(0);
   const Double_t kTimediffThreshold = 30 * MEG::nanosecond; // For PLCL-based initialisation

   if (GetSP()->GetFitInitMethod() < 0 ||
       GetSP()->GetFitInitMethod() == 0 ||
       GetSP()->GetFitInitMethod() / 1 % 10 > 0) {
      // Local weighted mean was taken before calling this method.
      // Nothing is done here.
   } else if (GetSP()->GetFitInitMethod() / 10 % 10 > 0) {
      auto pXECPL = gAnalyzer->GetXECPileupResult();
      if (!pXECPL) {
         Report(R_WARNING, "No XECPileup result is filled. Skip uv initialization by XECPL.");
         return;
      }
      nGamma = pXECPL->GetnpeakldInner();
      if (nGamma < 1) {
         return;
      }

      // Main gamma selection
      if (GetSP()->GetFitInitMethod() / 100 % 10 > 0) {
         // Distance to trigger patch
         auto patchid = gAnalyzer->GetTRGXECOnline()->GetPatchId();
         if (patchid > -1) FetchTRGPatchPos(uvwInit);

         Double_t mindist2patch = 1000;
         for (Short_t iGamma = 0; iGamma < nGamma; ++iGamma) {
            Double_t upeak = pXECPL->GetupeakldInnerAt(iGamma);
            Double_t vpeak = pXECPL->GetvpeakldInnerAt(iGamma);
            Double_t dist = TMath::Sqrt((upeak - uvwInit[0]) * (upeak - uvwInit[0]) +
                                        (vpeak - uvwInit[1]) * (vpeak - uvwInit[1]));
            if (dist < mindist2patch) {
               iMainGamma = iGamma;
               mindist2patch = dist;
            }
         } // end of gamma loop
      } else {
         // Largest energy
         Float_t Emaxgamma = 0; // Size of Patch in cm * 1.5
         for (Short_t iGamma = 0; iGamma < nGamma; ++iGamma) {
            auto epeak = pXECPL->GetepeakldInnerAt(iGamma);
            if (epeak > Emaxgamma) {
               Emaxgamma = epeak;
               iMainGamma = iGamma;
            }
         } // end of gamma loop
      } // Main gamma selection method

      if (iMainGamma > -1) {
         uvwInit[0] = pXECPL->GetupeakldInnerAt(iMainGamma);
         uvwInit[1] = pXECPL->GetvpeakldInnerAt(iMainGamma);
         uvwInit[2] = pXECPL->GetwpeakldInnerAt(iMainGamma);
      }
   } else if (GetSP()->GetFitInitMethod() / 100 % 10 > 0) {
      // TRG patch ID
      auto patchid = gAnalyzer->GetTRGXECOnline()->GetPatchId();
      if (patchid > -1) FetchTRGPatchPos(uvwInit);
   } else if (GetSP()->GetFitInitMethod() / 1000 % 10 > 0) {
      // PileupClustering
      if (!gAnalyzer->GetXECPileupClusteringResultSize()) {
         Report(R_WARNING, "No PileupClustering result is filled. Max-Npho-based initialization applied.");
         return;
      }
      Float_t maxNpho(-1);
      for (Short_t iGamma = 0; iGamma < gAnalyzer->GetXECPileupClusteringResultSize(); ++iGamma) {
         auto pPLCL = gAnalyzer->GetXECPileupClusteringResultAt(iGamma);
         if (pPLCL->Getmode() != 0) { // Skip outer spatial peaks
            continue;
         }
         // tpeak - PMT SumWF cftime
         auto tdiff = TMath::Abs(pPLCL->Gettpeak() - gAnalyzer->GetXECSumWaveform()->GetSumWaveformCftimeAt(1));
         if (tdiff < kTimediffThreshold) {
            auto clusterid = gAnalyzer->GetXECClusterInfo2At(pPLCL->Getpmpeak())->Getclusterid();
            auto npho = gAnalyzer->GetXECPMClusterAt(clusterid)->GetnphoAt(0);
            if (npho > maxNpho) {
               maxNpho = npho;
               iMainGamma = iGamma;
            }
         }
      } // end of PileupClustering index loop
      if (iMainGamma > -1) {
         auto pPLCL = gAnalyzer->GetXECPileupClusteringResultAt(iMainGamma);
         uvwInit[0] = pPLCL->Getupeak();
         uvwInit[1] = pPLCL->Getvpeak();
         uvwInit[2] = pPLCL->Getwpeak();
      }
   }

   if (fVerbose) {
      std::cout << "FitInitMethod: " << GetSP()->GetFitInitMethod() <<
                   ", uvwInit: " << uvwInit[0] << " " << uvwInit[1] << std::endl;
   }
}

//______________________________________________________________________________
void MEGTXECPosLocalFit::FetchTRGPatchPos(Double_t* uvwPatch)
{
   Int_t patchid = gAnalyzer->GetTRGXECOnline()->GetPatchId();
   Int_t crateid = patchid / 16;
   Int_t slotid = patchid - crateid * 16;
   const Int_t nPM = gAnalyzer->GetXECRunHeader()->GetNPM();
   Double_t uave = 0;
   Double_t vave = 0;
   Int_t nchannel = 0;
   for (Int_t iPM = 0; iPM < nPM; iPM++) {
      TString wdbname = gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetWDBname();
      if (wdbname.Contains(Form("MPPC%02d-%d", crateid, slotid))) {
         uave += gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetUVWAt(0);
         vave += gAnalyzer->GetXECPMRunHeaderAt(iPM)->GetUVWAt(1);
         nchannel++;
      }
   }
   uave /= nchannel;
   vave /= nchannel;
   uvwPatch[0] = uave;
   uvwPatch[1] = vave;
   uvwPatch[2] = 0.1;
}

void MEGTXECPosLocalFit::FillExpNphoHist(Int_t iRegion)
{
   MEGXECPoslMonitorPlots* poslPlots = (MEGXECPoslMonitorPlots*)gAnalyzer->GetXECPoslMonitorPlots();
   TObjArray* hist_posl = (TObjArray*)poslPlots->GetposlHists();
   TH2F* hNpho_exp = (TH2F*)hist_posl->FindObject(Form("fHNpho_exp%d", iRegion));
   hNpho_exp->Reset();
   for (Int_t iRow = 0; iRow < kNUMPM2[0]; iRow++) {
      for (Int_t iColumn = 0; iColumn < kNUMPM1[0]; iColumn++) {
         Int_t PM_this = iRow * kNUMPM1[0] + iColumn;
         if (PMArray[PM_this]->IsUsed4LSAF(iRegion)) {
            hNpho_exp->SetBinContent(iColumn + 1, iRow + 1, PMArray[PM_this]->GetNexp());
         }
      }
   }
}

//______________________________________________________________________________
Float_t MEGTXECPosLocalFit::GetMaximumSolidAngleToBadPM(Double_t* xyz)
{
   // Calculate solid angle from gamma-ray xyz position to the closest bad PM.
   // First calculate the distance between gamma and the closest bad MPPC (PMT) because the solid angle calculation costs computing time.
   // Return the larger solid angle either to the MPPC or the PMT.

   TVector3 gammaxyz(xyz[0], xyz[1], xyz[2]);

   // Calculate MPPC and PMT having the minimum distance to the gamma-ray xyz position
   Short_t iMinDistMPPC(-1), iMinDistPMT(-1);
   Float_t dist2(-1), mindist2mppc(1e10), mindist2pmt(1e10);
   for (Short_t iPM = 0; iPM < gAnalyzer->GetXECRunHeader()->GetNPM(); ++iPM) {
      if (PMArray[iPM]->GetIsBad()) {
         dist2 = (PMArray[iPM]->GetXYZ().X() - xyz[0]) * (PMArray[iPM]->GetXYZ().X() - xyz[0]) +
                 (PMArray[iPM]->GetXYZ().Y() - xyz[1]) * (PMArray[iPM]->GetXYZ().Y() - xyz[1]) +
                 (PMArray[iPM]->GetXYZ().Z() - xyz[2]) * (PMArray[iPM]->GetXYZ().Z() - xyz[2]);
         if (PMArray[iPM]->IsSiPM() && dist2 < mindist2mppc) {
            mindist2mppc = dist2;
            iMinDistMPPC = iPM;
         } else if (!PMArray[iPM]->IsSiPM() && dist2 < mindist2pmt) {
            mindist2pmt = dist2;
            iMinDistPMT = iPM;
         }
      }
   } // end of PM loop

   // Calculate solid angle
   Float_t solidAngle(-1000);
   if (iMinDistMPPC > -1 && iMinDistPMT > -1) {
      solidAngle = TMath::Max(PMSolidAngle::PMSolidAngleMPPC(gammaxyz, PMArray[iMinDistMPPC]->GetXYZ(), PMArray[iMinDistMPPC]->GetNorm()),
                              PMSolidAngle::PMSolidAnglePMT(gammaxyz, PMArray[iMinDistPMT]->GetXYZ(), PMArray[iMinDistPMT]->GetNorm()));
   } else if (iMinDistMPPC > -1) {
      solidAngle = PMSolidAngle::PMSolidAngleMPPC(gammaxyz, PMArray[iMinDistMPPC]->GetXYZ(), PMArray[iMinDistMPPC]->GetNorm());
   } else if (iMinDistPMT > -1) {
      solidAngle = PMSolidAngle::PMSolidAnglePMT(gammaxyz, PMArray[iMinDistPMT]->GetXYZ(), PMArray[iMinDistPMT]->GetNorm());
   }
   if (solidAngle >= 0) {
      solidAngle *= 4 * TMath::Pi(); // PMSolidAngleMPPC(PMT) returns the ratio to 4pi.
   }

   return solidAngle;
}

