// $Id$
#ifndef XECTOOLS_H
#define XECTOOLS_H

#include <Rtypes.h>
#include <TMath.h>
#include <TVector3.h>
#include <TSpline.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <Math/Rotation3D.h>
#include <Math/AxisAngle.h>
#include <Math/EulerAngles.h>
#include <Math/Vector3D.h>
#include <Math/Point3D.h>
#include <Math/Transform3D.h>
#include "ROMEiostream.h"
#include "units/MEGSystemOfUnits.h"

double comp_ellint_1(double);
double comp_ellint_3(double, double);

namespace XECTOOLS
{
/*
 * Before 2021, the definition of UVW was tied to XYZ and thus
 * not a true local coordinate system. Detector alignment would
 * shift the detector in UVW, which is not what we want.
 */
enum class UVWDefinition : Short_t {INVALID = -1, Global, Local};

// Rotation method
enum class rotationMethod : Short_t {Extrinsic, Intrinsic};


// externed XEC variables
extern Double_t gXERIN; // Radius at which the MPPCs are located
extern Double_t gXEROUT;
extern Double_t gXEZIN;
extern Double_t gXEZOUT;
extern Double_t gXEPHI;
extern Double_t gXETANTH;
extern Double_t gXECONEZ;
extern Double_t gXECOSTH;
extern Double_t gXECCornerPosition[8][3];
extern Double_t gXECADHOCSHIFT[3]; // Ad Hoc shift
extern Double_t gXECADHOCROTATION[3]; // Ad Hoc rotation, NOT USED
extern Double_t gXECVISHIFT[3]; // Shift to get from VI to current detector
extern Double_t gXECVIANGLES[3]; // Rotation to get from VI to current detector
extern XECTOOLS::UVWDefinition gXECUVWDEF;

// Distance btw/ surface of spacer (r=xerin) and MPPC photoelectroric surface
const Double_t kXECMPPCHeight = 1.3 * MEG::millimeter;

// Task ID to choose reconstruction to fill RecData
const Int_t kEnergyLinearFit     = 0;
const Int_t kEnergyMinuit        = 1;
const Int_t kEnergyEneTotalSum   = 2;
const Int_t kEnergyEneTotalSumPE = 3;
const Int_t kEnergyEneTotalSumPEEnhanced = 4;
const Int_t kEnergySumWaveform2PEEnhanced = 5; // combination of SumWaveform2 and EneTotalSumRec
const Int_t kEnergyFastRec = 6;

const Int_t kPositionLinearFit   = 0;
const Int_t kPositionMinuit      = 2;
const Int_t kPositionTimeMinuit  = 3;
const Int_t kPositionPosLocalFit = 4;

const Int_t kTimeWMean           = 1;
const Int_t kTimeTimeMinuit      = 3;
const Int_t kTimePosLocalFit     = 4;
const Int_t kTimeTimeFit         = 4;
const Int_t kTimeWeightedAverage = 5;

const Int_t kEvstatPLUF   = 0;
const Int_t kEvstatSumWF4 = 1;

// Pileup event status
const TString evstatName[5] = {"NoPL", "Unfolded", "Coinc", "DL", "NoConv"};

// Fiducial volume in MEG2
const Double_t kXECFiducialUMin = -23.9 * MEG::centimeter;
const Double_t kXECFiducialUMax =  23.9 * MEG::centimeter;
const Double_t kXECFiducialVMin = -67.9 * MEG::centimeter;
const Double_t kXECFiducialVMax =  67.9 * MEG::centimeter;
const Double_t kXECFiducialWMin =   0.0 * MEG::centimeter;
const Double_t kXECFiducialWMax =  42.0 * MEG::centimeter;

// Invalid reconstruction
const Double_t kXECInvalidEnergy    =   -10 * MEG::GeV;
const Double_t kXECInvalidTime      =  1e10 * MEG::second;
const Double_t kXECInvalidX         =    10 * MEG::meter; // this is positive to make direction invalid
const Double_t kXECInvalidY         =   -10 * MEG::meter;
const Double_t kXECInvalidZ         =   -10 * MEG::meter;
const Double_t kXECInvalidU         =   -10 * MEG::meter;
const Double_t kXECInvalidV         =   -10 * MEG::meter;
const Double_t kXECInvalidW         =   -10 * MEG::meter;
const Double_t kXECInvalidSigma2    =   -10 * MEG::meter * MEG::meter;

void InitXECGeometryParameters(UVWDefinition definition,
                               Double_t xerin, Double_t xerout, Double_t xezin, Double_t xezout,
                               Double_t xephi,
                               Double_t xshift = 0, Double_t yshift = 0, Double_t zshift = 0,
                               Double_t xrotate = 0, Double_t yrotate = 0, Double_t zrotate = 0);
void SetVITransformationParameters(Double_t xshift, Double_t yshift, Double_t zshift, Double_t phi, Double_t theta, Double_t psi);

void XYZ2RP(const Double_t *xyz, Double_t *rp);   // R : radius in the cylindrical polar coordinates
void XYZ2RPZ(const Double_t *xyz, Double_t *rpz); // R : radius in the cylindrical polar coordinates
template<class T> void XYZ2UVW(const Double_t*, Double_t*, T) = delete;
void XYZ2UVW(const Double_t *xyz, Double_t *uvw, bool adHocShift = false);
template<class T> void UVW2XYZ(const Double_t*, Double_t*, T) = delete;
void UVW2XYZ(const Double_t *uvw, Double_t *xyz, bool adHocShift = false);

// Fitting function
Double_t ExpGaus(Double_t *x, Double_t *par);
Double_t WExpGaus(Double_t *x, Double_t *par);
Double_t EGammaFunc(Double_t *x, Double_t *par);
Double_t Poisson(Double_t *x, Double_t *par);
Double_t PoissonGaus(Double_t *x, Double_t *par);
Double_t LogPois(Double_t *x, Double_t *par);
Double_t LogPoisG(Double_t *x, Double_t *par);
Double_t XECWaveformFncBase0(Double_t *x, Double_t *par);
Double_t XECWaveformFncBase1(Double_t *x, Double_t *par);
Double_t XECSciWaveformFnc(Double_t *x, Double_t *par);
Double_t XECWaveformFnc(Double_t *x, Double_t *par);
Double_t XECDRSWaveformFnc(Double_t *x, Double_t *par);
Double_t XECTRGWaveformFnc(Double_t *x, Double_t *par);
Double_t XECDRSWaveformDoubleFnc(Double_t *x, Double_t *par);

Bool_t IsInShadow(const Double_t *src, const Double_t *dest);
Bool_t IsInShadow(const TVector3 src, const TVector3 dest);

inline void CheckInitialized()
{
   // Print warning if parameters are not initialized.
   if (gXECUVWDEF == UVWDefinition::INVALID) {
      Report(R_WARNING, "Local coordinate system is not defined. Please call XECTOOLS::InitXECGeometryPatameters");
   }
   if (TMath::IsNaN(gXERIN)) {
      Report(R_WARNING, "XeRIN = 0. Probably XECTOOLS not inittialized. Please call XECTOOLS::InitXECGeometryParameters");
   }
   if (TMath::IsNaN(gXECVISHIFT[0])) {
      Report(R_WARNING, "VI transformation parameters are not initialized. Please call XECTOOLS::SetVITransformationParameters");
   }
}

// Convert XYZ to where it would be in the virtual ideal
// Using a template allows to use either Vector3D or Point3D.
// Note that the rotation seems to be defined as inverse. For some reason,
// ROOT appears to implement rotations the other way around. If someone understands,
// please tell Lukas
// Note that depending on the type, the translation has no effect:
// Vectors can only be rotated while points can be shifted.
template <typename T>
T XYZ2VI(const T& xyz)
{
   CheckInitialized();
   ROOT::Math::EulerAngles rot{gXECVIANGLES[0] * TMath::DegToRad(),
                               gXECVIANGLES[1] * TMath::DegToRad(),
                               gXECVIANGLES[2] * TMath::DegToRad()};
   ROOT::Math::Translation3D trans{ gXECVISHIFT[0], gXECVISHIFT[1], gXECVISHIFT[2] };

   return rot * trans.Inverse() * xyz;
}

template <typename T>
T VI2XYZ(const T& xyz)
{
   CheckInitialized();
   ROOT::Math::EulerAngles rot{gXECVIANGLES[0] * TMath::DegToRad(),
                               gXECVIANGLES[1] * TMath::DegToRad(),
                               gXECVIANGLES[2] * TMath::DegToRad()};
   ROOT::Math::Translation3D trans{ gXECVISHIFT[0], gXECVISHIFT[1], gXECVISHIFT[2] };

   return trans * rot.Inverse() * xyz;
}

inline void applyAdHocRotation(Double_t &x, Double_t &y, Double_t &z, Bool_t forward)
{
   CheckInitialized();
   TVector3 xyz(x, y, z);
   if (forward) {
      xyz.RotateZ(gXECADHOCROTATION[2] * TMath::DegToRad());
      xyz.RotateY(gXECADHOCROTATION[1] * TMath::DegToRad());
      xyz.RotateX(gXECADHOCROTATION[0] * TMath::DegToRad());
   } else {
      xyz.RotateX(-gXECADHOCROTATION[0] * TMath::DegToRad());
      xyz.RotateY(-gXECADHOCROTATION[1] * TMath::DegToRad());
      xyz.RotateZ(-gXECADHOCROTATION[2] * TMath::DegToRad());
   }

   x = xyz[0];
   y = xyz[1];
   z = xyz[2];
}
// Alias for backward compatibility
const auto XYZRotation = applyAdHocRotation;

// Apply ad-hoc shift
inline void applyAdHocShift(Double_t &x, Double_t &y, Double_t &z, Bool_t forward)
{
   // If forward is true, x is shifted to x+dx
   CheckInitialized();
   if (forward) {
      x = x + gXECADHOCSHIFT[0];
      y = y + gXECADHOCSHIFT[1];
      z = z + gXECADHOCSHIFT[2];
   } else {
      x = x - gXECADHOCSHIFT[0];
      y = y - gXECADHOCSHIFT[1];
      z = z - gXECADHOCSHIFT[2];
   }
}
// Alias for backward compatibility
const auto XYZShift = applyAdHocShift;

[[nodiscard]] inline Double_t XYZ2R(Double_t x, Double_t y, Double_t /*z*/)
{
   // R : radius in the cylindrical polar coordinates
   return TMath::Sqrt(x * x + y * y);
}

[[nodiscard]] inline Double_t XYZ2Perp(Double_t x, Double_t y, Double_t /*z*/)
{
   return TMath::Sqrt(x * x + y * y);
}

[[nodiscard]] inline Double_t XYZ2Theta(Double_t x, Double_t y, Double_t z)
{
   return x == 0.0 && y == 0.0 && z == 0.0 ? 0.0 : TMath::ATan2(XYZ2R(x, y, z), z) * MEG::radian;
}

[[nodiscard]] inline Double_t XYZ2Phi(Double_t x, Double_t y, Double_t /*z*/)
{
   return x == 0.0 && y == 0.0 ? 0.0 : 180 * MEG::degree + TMath::ATan2(-y, -x) * MEG::radian;
}

template<class T> void XYZ2U(Double_t, Double_t, Double_t, T) = delete;
[[nodiscard]] inline Double_t XYZ2U(Double_t x, Double_t y, Double_t z, bool adHocShift = false)
{
   CheckInitialized();
   if (adHocShift) {
      applyAdHocShift(x, y, z, false);
      applyAdHocRotation(x, y, z, false);
   }
   if (gXECUVWDEF == UVWDefinition::Local) {
      ROOT::Math::XYZPoint p(x, y, z);
      p = XYZ2VI(p);
      z = p.z();
   }
   return z;
}

template<class T> void XYZ2V(Double_t, Double_t, Double_t, T) = delete;
[[nodiscard]] inline Double_t XYZ2V(Double_t x, Double_t y, Double_t z, bool adHocShift = false)
{
   CheckInitialized();
   if (adHocShift) {
      applyAdHocShift(x, y, z, false);
      applyAdHocRotation(x, y, z, false);
   }
   if (gXECUVWDEF == UVWDefinition::Local) {
      ROOT::Math::XYZPoint p(x, y, z);
      p = XYZ2VI(p);
      x = p.x();
      y = p.y();
      z = p.z();
   }
   return -(XYZ2Phi(x, y, z) - 180 * MEG::degree) / MEG::radian * (gXERIN + kXECMPPCHeight);
}

template<class T> void XYZ2W(Double_t, Double_t, Double_t, T) = delete;
[[nodiscard]] inline Double_t XYZ2W(Double_t x, Double_t y, Double_t z, bool adHocShift = false)
{
   CheckInitialized();
   if (adHocShift) {
      applyAdHocShift(x, y, z, false);
      applyAdHocRotation(x, y, z, false);
   }
   if (gXECUVWDEF == UVWDefinition::Local) {
      ROOT::Math::XYZPoint p(x, y, z);
      p = XYZ2VI(p);
      x = p.x();
      y = p.y();
      z = p.z();
   }
   return XYZ2R(x, y, z) - (gXERIN + kXECMPPCHeight);
}

template<class T> void UVW2X(Double_t, Double_t, Double_t, T) = delete;
// Tip: if you need x, y and z, call UVW2XYZ instead
[[nodiscard]] inline Double_t UVW2X(Double_t u, Double_t v, Double_t w, bool adHocShift = false)
{
   CheckInitialized();
   Double_t xyz[3];
   Double_t uvw[3] = {u, v, w};
   UVW2XYZ(uvw, xyz, adHocShift);
   return xyz[0];
}

template<class T> void UVW2Y(Double_t, Double_t, Double_t, T) = delete;
// Tip: if you need x, y and z, call UVW2XYZ instead
[[nodiscard]] inline Double_t UVW2Y(Double_t u, Double_t v, Double_t w, bool adHocShift = false)
{
   CheckInitialized();
   Double_t xyz[3];
   Double_t uvw[3] = {u, v, w};
   UVW2XYZ(uvw, xyz, adHocShift);
   return xyz[1];
}

template<class T> void UVW2Z(Double_t, Double_t, Double_t, T) = delete;
// Tip: if you need x, y and z, call UVW2XYZ instead
[[nodiscard]] inline Double_t UVW2Z(Double_t u, Double_t v, Double_t w, bool adHocShift = false)
{
   CheckInitialized();
   Double_t xyz[3];
   Double_t uvw[3] = {u, v, w};
   UVW2XYZ(uvw, xyz, adHocShift);
   return xyz[2];
}

[[nodiscard]] inline Double_t CosOpeningAngle(Double_t theta0, Double_t phi0, Double_t theta1,
                                              Double_t phi1) // All angles in radian
{
   return TMath::Sin(theta0) * TMath::Cos(phi0) * TMath::Sin(theta1) * TMath::Cos(phi1)
          + TMath::Sin(theta0) * TMath::Sin(phi0) * TMath::Sin(theta1) * TMath::Sin(phi1)
          + TMath::Cos(theta0) * TMath::Cos(theta1);
}

[[nodiscard]] inline Double_t PMTU(Double_t u)
{
   return (u + 3.1 * 101) - (static_cast<Int_t>((u + 3.1 * 101) / 6.2) + 0.5) * 6.2;
}
[[nodiscard]] inline Double_t PMTV(Double_t v)
{
   return (v + 3.1 * 100) - (static_cast<Int_t>((v + 3.1 * 100) / 6.2) + 0.5) * 6.2;
}

void GammaRayCrossingUVW(const Double_t *vertex, Double_t costh, Double_t phi, Double_t r, Double_t *uvw);

Double_t GetHistCorrection(Int_t method, Double_t x, TH1D *hist, TSpline *sp = 0);
Double_t GetHistCorrection2D(Int_t method, Double_t x, TH1D **histx, Int_t nx, Double_t y, TH1D **histy,
                             Int_t ny);
Double_t GetHistCorrection2D(Int_t method, Double_t x, Double_t y, TH2D *hist);
Double_t GetHistCorrection3D(Int_t method, Double_t x, Double_t y, Double_t z, TH3D *hist);
Double_t LinearInterpolate(Double_t x, Double_t *vref, Double_t x1, Double_t x2);
Double_t LinearInterpolate2D(Double_t x, Double_t y, Double_t *vref,
                             Double_t xlb, Double_t ylb, Double_t dx, Double_t dy);
}

#endif                          //XECTOOLS_H
