// $Id$
#include <math.h>
#include "xec/xectools.h"
#if defined( R__VISUAL_CPLUSPLUS )
#   pragma warning( push )
#   pragma warning( disable : 4800 )
#endif                          // R__VISUAL_CPLUSPLUS
#include <Riostream.h>
#include <TF1.h>
#include <TVector3.h>
#if defined( R__VISUAL_CPLUSPLUS )
#   pragma warning( pop )
#endif                          // R__VISUAL_CPLUSPLUS

// externed XEC variables
Double_t XECTOOLS::gXERIN  = TMath::SignalingNaN();
Double_t XECTOOLS::gXEROUT = TMath::SignalingNaN();
Double_t XECTOOLS::gXEZIN  = TMath::SignalingNaN();
Double_t XECTOOLS::gXEZOUT = TMath::SignalingNaN();
Double_t XECTOOLS::gXEPHI  = TMath::SignalingNaN();
Double_t XECTOOLS::gXETANTH;
Double_t XECTOOLS::gXECONEZ;
Double_t XECTOOLS::gXECOSTH;
Double_t XECTOOLS::gXECCornerPosition[8][3] = {{0}};
Double_t XECTOOLS::gXECADHOCSHIFT[3]    = {TMath::SignalingNaN()};
Double_t XECTOOLS::gXECADHOCROTATION[3] = {TMath::SignalingNaN()};
Double_t XECTOOLS::gXECVISHIFT[3]       = {TMath::SignalingNaN()}; // Shift to get from VI to current detector
Double_t XECTOOLS::gXECVIANGLES[3]      = {TMath::SignalingNaN()}; // Rotation to get from VI to current detector
XECTOOLS::UVWDefinition XECTOOLS::gXECUVWDEF = XECTOOLS::UVWDefinition::INVALID;

//______________________________________________________________________________
void XECTOOLS::InitXECGeometryParameters(UVWDefinition definition,
                                         Double_t xerin, Double_t xerout, Double_t xezin, Double_t xezout,
                                         Double_t xephi,
                                         Double_t xadhocshift, Double_t yadhocshift, Double_t zadhocshift,
                                         Double_t xadhocrotate, Double_t yadhocrotate, Double_t zadhocrotate)
{
   gXECUVWDEF = definition;
   gXERIN  = xerin;
   gXEROUT = xerout;
   gXEZIN  = xezin;
   gXEZOUT = xezout;
   gXEPHI  = xephi;

   gXETANTH = 0.5 * (gXEZOUT - gXEZIN) / (gXEROUT - gXERIN);
   gXECONEZ = gXEZIN / 2 - gXERIN * gXETANTH;
   gXECOSTH = (gXEROUT - gXERIN) / TMath::Sqrt(0.5 * (gXEZOUT - gXEZIN) * 0.5 * (gXEZOUT - gXEZIN)
                                               + (gXEROUT - gXERIN) * (gXEROUT - gXERIN));
   gXECCornerPosition[0][0] = -gXEROUT * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[0][1] = +gXEROUT * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[0][2] = +gXEZOUT / 2;
   gXECCornerPosition[1][0] = -gXERIN * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[1][1] = +gXERIN * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[1][2] = +gXEZIN / 2;
   gXECCornerPosition[2][0] = -gXERIN * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[2][1] = +gXERIN * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[2][2] = -gXEZIN / 2;
   gXECCornerPosition[3][0] = -gXEROUT * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[3][1] = +gXEROUT * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[3][2] = -gXEZOUT / 2;
   gXECCornerPosition[4][0] = -gXEROUT * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[4][1] = -gXEROUT * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[4][2] = +gXEZOUT / 2;
   gXECCornerPosition[5][0] = -gXERIN * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[5][1] = -gXERIN * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[5][2] = +gXEZIN / 2;
   gXECCornerPosition[6][0] = -gXERIN * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[6][1] = -gXERIN * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[6][2] = -gXEZIN / 2;
   gXECCornerPosition[7][0] = -gXEROUT * TMath::Cos(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[7][1] = -gXEROUT * TMath::Sin(gXEPHI * TMath::DegToRad() / 2);
   gXECCornerPosition[7][2] = -gXEZOUT / 2;
   gXECADHOCSHIFT[0] = xadhocshift;
   gXECADHOCSHIFT[1] = yadhocshift;
   gXECADHOCSHIFT[2] = zadhocshift;
   gXECADHOCROTATION[0] = xadhocrotate;
   gXECADHOCROTATION[1] = yadhocrotate;
   gXECADHOCROTATION[2] = zadhocrotate;
}

//______________________________________________________________________________
void XECTOOLS::SetVITransformationParameters(Double_t xshift, Double_t yshift, Double_t zshift,
                                             Double_t phi, Double_t theta, Double_t psi)
{
   gXECVISHIFT[0] = xshift;
   gXECVISHIFT[1] = yshift;
   gXECVISHIFT[2] = zshift;
   gXECVIANGLES[0] = phi;
   gXECVIANGLES[1] = theta;
   gXECVIANGLES[2] = psi;
}

//______________________________________________________________________________
Double_t XECTOOLS::ExpGaus(Double_t *x, Double_t *par)
{
   //http://pibeta.phys.virginia.edu/~pibeta/docs/publications/penny_diss/node35.html
   //par[0] : height of Gaussian
   //par[1] : peak position
   //par[2] : sigma of Gaussian
   //par[3] : transition point between gaussian and exponential

   Double_t fitval;

   if (x[0] > par[1] + par[3]) {
      if (par[2] != 0) {
         fitval = par[0] * TMath::Exp(-1 * (x[0] - par[1]) * (x[0] - par[1]) / 2 / par[2] / par[2]);
      } else {
         fitval = 0;
      }
   } else {
      if (par[2] != 0) {
         fitval = par[0] * TMath::Exp(par[3] / par[2] / par[2] * (par[3] / 2 - (x[0] - par[1])));
      } else {
         fitval = 0;
      }
   }

   return fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::WExpGaus(Double_t *x, Double_t *par)
{
   //par[0] : height of Gaussian, 1st ExpGaus
   //par[1] : peak position, 1st ExpGaus
   //par[2] : sigma of Gaussian, 1st ExpGaus
   //par[3] : transition point between gaussian and exponential, 1st ExpGaus
   //par[4] : height of Gaussian, 2nd ExpGaus
   //par[5] : peak position, 2nd ExpGaus
   //par[6] : sigma of Gaussian, 2nd ExpGaus
   //par[7] : transition point between gaussian and exponential, 2nd ExpGaus

   Double_t fitval;

   // 1st ExpGaus
   if (x[0] > par[1] + par[3]) {
      if (par[2] != 0) {
         fitval = par[0] * TMath::Exp(-1 * (x[0] - par[1]) * (x[0] - par[1]) / 2 / par[2] / par[2]);
      } else {
         fitval = 0;
      }
   } else {
      if (par[2] != 0) {
         fitval = par[0] * TMath::Exp(par[3] / par[2] / par[2] * (par[3] / 2 - (x[0] - par[1])));
      } else {
         fitval = 0;
      }
   }

   // 2nd ExpGaus
   if (x[0] > par[5] + par[7]) {
      if (par[6] != 0) {
         fitval += par[4] * TMath::Exp(-1 * (x[0] - par[5]) * (x[0] - par[5]) / 2 / par[6] / par[6]);
      } else {
         fitval += 0;
      }
   } else {
      if (par[6] != 0) {
         fitval += par[4] * TMath::Exp(par[7] / par[6] / par[6] * (par[7] / 2 - (x[0] - par[5])));
      } else {
         fitval += 0;
      }
   }

   return fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::EGammaFunc(Double_t *x, Double_t *par)
{
   //par[0] : Gaussian Height
   //par[1] : Mean
   //par[2] : Sigma of Gaussian
   //par[3] : Exponential constant
   //par[4] : Exponential Height
   //par[5] : Bin width (should be fixed in fitting).

   if (par[2] == 0) {
      return 0;
   }

   Double_t y;
   Double_t val = 0;
   Double_t width = TMath::Abs(par[5]);

#if 0
   for (y = x[0] - 10 * par[2]; y < x[0] + 10 * par[2] && y < par[1]; y += width) {
      val += width * par[4] * TMath::Exp((y - par[1]) / par[3]) * TMath::Gaus(x[0], y, par[2], kTRUE);
   }
#else
   Int_t i;
   Int_t halfWidth = static_cast<Int_t>(10 * par[2] / width);
   for (i = 0; i < 2 * halfWidth; i++) {
      y = (static_cast<Int_t>(x[0] / width) - halfWidth + i) * width;
      if (y > par[1]) {
         break;
      }
      val += width * par[4] * TMath::Exp((y - par[1]) / par[3]) * TMath::Gaus(x[0], y, par[2], kTRUE);
   }
#endif
   val += par[0] * TMath::Gaus(x[0], par[1], par[2], kFALSE);

   return val;
}

//______________________________________________________________________________
Double_t XECTOOLS::Poisson(Double_t *x, Double_t *par)
{
   // This function returns Poisson((x - par[2])
   // par[0] : constant
   // par[1] : mean   (must be limited greater than 0)
   // par[2] : pedestal

   Double_t X = x[0] - par[2];

   if (X < 0) {
      return 0;
   }

   return par[0] * TMath::Exp(-par[1]) * TMath::Power(par[1], X) / TMath::Gamma(X + 1);
}

//______________________________________________________________________________
Double_t XECTOOLS::PoissonGaus(Double_t *x, Double_t *par)
{
   // This function returns Poisson(x - par[2]) smeared by a Gaussian
   //   with sigma=par[3].
   // Smearing is done from 0 to 50 of x - par[2].

   // par[0] : constant
   // par[1] : mean   (must be limited greater than 0)
   // par[2] : pedestal
   // par[3] : smear

   if (par[3] <= 0) {
      return Poisson(x, par);
   }

   Double_t fitval = 0;
   int i;
   TF1 ga("ga", "gaus", -50, 50);
   ga.SetParameter(0, 1 / TMath::Sqrt(2 * TMath::Pi() * par[3]));
   ga.SetParameter(2, par[3]);
   Double_t X = x[0] - par[2];
   for (i = 0; i < 50; i++) {
      ga.SetParameter(1, i);
      fitval += par[0] * TMath::Exp(-par[1]) * TMath::Power(par[1], i) / (TMath::Gamma(i + 1)) * ga.Eval(X);
   }
   return fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::LogPois(Double_t *x, Double_t *par)
{
   Double_t logfatt, res, argomento;
   Int_t n;

   n = static_cast<Int_t>(x[0]);

   if (n < 0) {
      return 0.;
   } else if (n == 0) {
      res = par[0] * TMath::Exp(-par[1]);
      return res;
   } else {
      if (par[1] <= 30.) {
         logfatt = 0.;
         for (Int_t i = 1; i < n + 1; i++) {
            argomento = par[1] / static_cast<Float_t>(i) > 0 ? par[1] / static_cast<Float_t>(i) : 1e-7;
            logfatt = logfatt + TMath::Log(argomento);
         }
         par[2] = TMath::Sqrt(TMath::Abs(par[1]));
         res = par[0] * TMath::Exp(-par[1] + logfatt);
         return res;
      } else {
         par[2] = TMath::Sqrt(TMath::Abs(par[1]));
         argomento = par[1] > 0 ? par[1] : 1e-7;
         res = par[0] / TMath::Sqrt(2 * TMath::Pi() * argomento) *
               TMath::Exp(-0.5 * TMath::Power((n - argomento), 2) / argomento);
         return res;
      }
   }
}

//______________________________________________________________________________
Double_t XECTOOLS::LogPoisG(Double_t *x, Double_t *par)
{
   Double_t logsum = 0;
   Double_t res, argomento;
   Int_t i;

   if (par[1] <= 5) {
      for (i = 1; i < 81; i++) {
         Double_t x1[1] = {static_cast<Double_t>(i - 1)};
         argomento = par[2] > 0 ? par[2] * par[2] : 1e-7;
         logsum = logsum + LogPois(x1, par) * 1 / TMath::Sqrt(2 * TMath::Pi() * argomento) *
                  TMath::Exp(-0.5 * TMath::Power((x[0] - static_cast<Float_t>(i - 1.)), 2) / argomento);
      }
      par[2] = TMath::Sqrt(TMath::Abs(par[1]));
      return logsum;
   } else {
      par[2] = TMath::Sqrt(TMath::Abs(par[1]));
      res = LogPois(x, par);
      return res;
   }
}

//______________________________________________________________________________
Double_t XECTOOLS::XECWaveformFncBase0(Double_t *x, Double_t *par)
{
   //Simple waveform with 1 decay constant and 1 leading constant
   //8 Parameters and typical values
   //0:Offset voltage          0 [mV]
   //1:Leading time            -550--500 [nsec]
   //2:Leading time constant   [nsec]
   //3:Decay time constant     [nsec]
   //4:Amplitude               [mV]
   //5:Ratio of decay component
   //6:Time unit               1 [nsec]      [Fixed]
   //7:Satulation voltage      950 [mV]      [Fixed]
   //Positive amplitude returns negative pulse
   Double_t fitval     = 0;
   Double_t arg = x[0] - par[1];//leading time     [timeunit]
   Double_t tau        = par[2];//leading constant [timeunit]
   Double_t taudecay   = par[3];//decay constant   [timeunit]
   Double_t amplitude  = par[4];//amplitude
   Double_t ratio      = par[5];//Decay component
   Double_t timeunit   = par[6];//Scale of x
   Double_t satulation = par[7];//Satulation voltage

   if (arg > 0) {
      fitval += (1 - ratio) * TMath::Exp(-arg / (tau * timeunit));
      fitval += ratio * TMath::Exp(-arg / (taudecay * timeunit));
      fitval *= amplitude;//Normalize here if needed
   }
   fitval    += par[0];//+offset
   return TMath::Abs(fitval) > TMath::Abs(satulation)
          ? (fitval > 0 ? TMath::Abs(satulation) : -1 * TMath::Abs(satulation))
          : fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::XECWaveformFncBase1(Double_t *x, Double_t *par)
{
   //Simple waveform with 2 decay constant and 1 leading constant
   //8 Parameters and typical values
   //0:Offset voltage          0 [mV]
   //1:Leading time            -550--500 [nsec]
   //2:Leading time constant   45 [nsec]
   //3:Decay time constant     0.3*amplitude [mV]
   //4:Amplitude               [mV]
   //5:Ratio of decay component
   //6:Time unit               1 [nsec]      [Fixed]
   //7:Satulation voltage      950 [mV]      [Fixed]
   //Positive amplitude returns negative pulse
   Double_t fitval     = par[0];//amplitude + offset
   Double_t arg = x[0] - par[1];//leading time     [timeunit]
   Double_t tau        = par[2];//leading constant [timeunit]
   Double_t taudecay1  = par[3];//decay constant   [timeunit]
   Double_t taudecay2  = par[4];//decay constant   [timeunit]
   Double_t amplitude  = par[5];//amplitude
   Double_t ratio1     = par[6];//Decay component
   Double_t ratio2     = par[7];//Decay component
   Double_t timeunit   = par[8];//Scale of x
   Double_t satulation = par[9];//Satulation voltage

   if (arg > 0) {
      fitval += amplitude * (1 - ratio1 - ratio2) * TMath::Exp(-arg / (tau * timeunit));
      fitval += amplitude * (ratio1 * TMath::Exp(-arg / (taudecay1 * timeunit))
                             + ratio2 * TMath::Exp(-arg / (taudecay2 * timeunit)));
   }
   return TMath::Abs(fitval) > TMath::Abs(satulation)
          ? (fitval > 0 ? TMath::Abs(satulation) : -1 * TMath::Abs(satulation))
          : fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::XECSciWaveformFnc(Double_t *x, Double_t *par)
{
   //Simple waveform from xenon scintillation light.
   //No RC constant of PMT, no reflection and no electronic coupling
   //8 Parameters and typical values
   //0:Offset voltage          0 [mV]
   //1:Leading time            -550--500 [nsec]
   //2:Leading time constant   45 [nsec]
   //3:Factor of fast decay    1.3*amplitude [mV] [gamma]
   //4:Factor of slow decay    0.3*amplitude [mV] [gamma]
   //5:Factor of recombination 67.*amplitude [mV] [gamma]
   //6:Time unit               1 [nsec]      [Fixed]
   //7:Satulation voltage      950 [mV]      [Fixed]
   //Positive amplitude returns negative pulse
   Double_t fitval     = par[0];//amplitude + offset
   Double_t arg = x[0] - par[1];//leading time     [timeunit]
   Double_t tau        = par[2];//leading constant [timeunit]
   Double_t timeunit   = par[6];//Scale of x
   Double_t satulation = par[7];//Satulation voltage
   //Decay time constant
   Double_t a_fast = par[3],         a_slow = par[4],        a_recomb = par[5];
   Double_t tau_fast = 4.2 * timeunit, tau_slow = 22 * timeunit, tau_recomb = 45 * timeunit;

   if (arg > 0) {
      fitval += (-a_fast - a_slow - a_recomb) * TMath::Exp(-arg / (tau * timeunit));
      fitval += a_fast * TMath::Exp(-arg / tau_fast)
                + a_slow * TMath::Exp(-arg / tau_slow)
                + a_recomb * TMath::Exp(-arg / tau_recomb);
   }
   return TMath::Abs(fitval) > TMath::Abs(satulation)
          ? (fitval > 0 ? TMath::Abs(satulation) : -1 * TMath::Abs(satulation))
          : fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::XECWaveformFnc(Double_t *x, Double_t *par)
{
   //TEST IMPLIMENTATION. MODIFY IT.
   Double_t tau1, tau2;
   Double_t timeunit   = par[5];//
   Double_t argdiff    = par[4];//80*timeunit;
   Double_t satulation = par[6];//950
   Double_t tau_d = 2500 * timeunit, tau_i1 = 15 * timeunit, tau_i2 = 30 * timeunit;
   Double_t fitval, arg, arg2;

   arg = x[0] - par[1];
   arg2 = arg - argdiff;
   tau1 = par[2];//60*timeunit
   tau2 = 3.*tau1;//3*60*timeunit
   Double_t v1 = par[3];
   Double_t v2 = 0.05 * v1;
   fitval = par[0];

   Double_t ai1 = tau_d * tau1 / (tau_d - tau_i1) / (tau1 - tau_i1);
   Double_t a11 = -ai1;

   Double_t ai2 = tau_d * tau2 / (tau_d - tau_i2) / (tau2 - tau_i2);
   Double_t a22 = -ai2;

   if (arg > 0) {
      fitval += v1 * (ai1 * TMath::Exp(-arg / tau_i1) + a11 * TMath::Exp(-arg / tau1));
   }
   if (arg2 > 0 && argdiff != 0) {
      fitval += v2 * (ai2 * TMath::Exp(-arg2 / tau_i2) + a22 * TMath::Exp(-arg2 / tau2));
   }
   return TMath::Abs(fitval) > TMath::Abs(satulation)
          ? (fitval > 0 ? TMath::Abs(satulation) : -1 * TMath::Abs(satulation))
          : fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::XECTRGWaveformFnc(Double_t *x, Double_t *par)
{
   Double_t tau1, tau2;
   Double_t timeunit   = par[5];//
   Double_t argdiff    = par[4];//80*timeunit;
   Double_t satulation = par[6];//950
   Double_t tau_d = 2500 * timeunit, tau_i1 = 15 * timeunit, tau_i2 = 30 * timeunit;
   Double_t fitval, arg, arg2;

   arg = x[0] - par[1];
   arg2 = arg - argdiff;
   tau1 = par[2];
   tau2 = 3.*tau1;
   Double_t v1 = par[3];
   Double_t v2 = 0.05 * v1;
   fitval = par[0];

   Double_t ad1 = tau_d * tau1 / (tau_i1 - tau_d) / (tau1 - tau_d);
   Double_t ai1 = tau_d * tau1 / (tau_d - tau_i1) / (tau1 - tau_i1);
   Double_t a11 = tau_d * tau1 / (tau_d - tau1) / (tau_i1 - tau1);

   Double_t ad2 = tau_d * tau2 / (tau_i2 - tau_d) / (tau2 - tau_d);
   Double_t ai2 = tau_d * tau2 / (tau_d - tau_i2) / (tau2 - tau_i2);
   Double_t a22 = tau_d * tau2 / (tau_d - tau2) / (tau_i2 - tau2);

   if (arg > 0) {
      fitval += v1 * (ad1 * TMath::Exp(-arg / tau_d) + ai1 * TMath::Exp(-arg / tau_i1) + a11 * TMath::Exp(
                         -arg / tau1));
   }
   if (arg2 > 0 && argdiff != 0) {
      fitval += v2 * (ad2 * TMath::Exp(-arg2 / tau_d) + ai2 * TMath::Exp(-arg2 / tau_i2) + a22 * TMath::Exp(
                         -arg2 / tau2));
   }
   return TMath::Abs(fitval) > TMath::Abs(satulation)
          ? (fitval > 0 ? TMath::Abs(satulation) : -1 * TMath::Abs(satulation))
          : fitval;
}

//______________________________________________________________________________
Double_t XECTOOLS::XECDRSWaveformFnc(Double_t *x, Double_t *par)
{
   //TEMPORARY
   return XECWaveformFnc(x, par);
   //return XECSciWaveformFnc(x, par);
}

//______________________________________________________________________________
Double_t XECTOOLS::XECDRSWaveformDoubleFnc(Double_t *x, Double_t *par)
{
   //TEMPORARY
   const Int_t parnum = 6;
   return XECWaveformFnc(x, par) + XECWaveformFnc(x + parnum, par + parnum);
}

//______________________________________________________________________________
void XECTOOLS::XYZ2RP(const Double_t *xyz, Double_t *rp)
{
   // R : radius in the cylindrical polar coordinates
   rp[0] = XYZ2R(xyz[0], xyz[1], xyz[2]);
   rp[1] = XYZ2Phi(xyz[0], xyz[1], xyz[2]);
   return;
}

//______________________________________________________________________________
void XECTOOLS::XYZ2RPZ(const Double_t *xyz, Double_t *rpz)
{
   // R : radius in the cylindrical polar coordinates
   rpz[0] = XYZ2R(xyz[0], xyz[1], xyz[2]);
   rpz[1] = XYZ2Phi(xyz[0], xyz[1], xyz[2]);
   rpz[2] = xyz[2];
   return;
}

//______________________________________________________________________________
void XECTOOLS::XYZ2UVW(const Double_t *xyz, Double_t *uvw, bool applyAdHocShift)
{
   uvw[0] = XYZ2U(xyz[0], xyz[1], xyz[2], applyAdHocShift);
   uvw[1] = XYZ2V(xyz[0], xyz[1], xyz[2], applyAdHocShift);
   uvw[2] = XYZ2W(xyz[0], xyz[1], xyz[2], applyAdHocShift);
   return;
}

//______________________________________________________________________________
void XECTOOLS::UVW2XYZ(const Double_t *uvw, Double_t *xyz, bool adHocShift)
{
   CheckInitialized();
   xyz[0] = -1. * (uvw[2] + gXERIN + kXECMPPCHeight) * TMath::Cos(uvw[1] / (gXERIN + kXECMPPCHeight));
   xyz[1] = (uvw[2] + gXERIN + kXECMPPCHeight) * TMath::Sin(uvw[1] / (gXERIN + kXECMPPCHeight));
   xyz[2] = uvw[0];

   if (gXECUVWDEF == UVWDefinition::Local) {
      ROOT::Math::XYZPoint p(xyz[0], xyz[1], xyz[2]);
      p = VI2XYZ(p);
      xyz[0] = p.x();
      xyz[1] = p.y();
      xyz[2] = p.z();
   }
   if (adHocShift) {
      applyAdHocRotation(xyz[0], xyz[1], xyz[2], true);
      applyAdHocShift(xyz[0], xyz[1], xyz[2], true);
   }
   return;
}

//______________________________________________________________________________
Bool_t XECTOOLS::IsInShadow(const Double_t *src, const Double_t *dest)
{
   // check if two points can be directly connected by a linear function
   // in the calorimeter.
   // parameters have to be initialized by InitXECGeometryParameters before calling this.

   // Two points are on a line parallel to the z axis.
   if (src[0] == dest[0] && src[1] == dest[1]) {
      return kFALSE;
   }

   // "src" or "dest" is inner than inner face.
   if (src[0] * src[0] + src[1] * src[1] < (gXERIN * gXERIN) ||
       dest[0] * dest[0] + dest[1] * dest[1] < (gXERIN * gXERIN)) {
      return kTRUE;
   }

   // no intersection with inner face
   if ((dest[0] * src[1] - dest[1] * src[0]) * (dest[0] * src[1] - dest[1] * src[0]) /
       ((dest[0] - src[0]) * (dest[0] - src[0]) + (dest[1] - src[1]) * (dest[1] - src[1])) >= (gXERIN * gXERIN)) {
      return kFALSE;
   }

   // check if closest point to the origine from the line is between "src" and "dest"
   Double_t a = (src[0] * (src[0] - dest[0]) + src[1] * (src[1] - dest[1])) /
                ((dest[0] - src[0]) * (dest[0] - src[0]) + (dest[1] - src[1]) * (dest[1] - src[1]));

   return (a > 0 && a < 1);
}

//______________________________________________________________________________
Bool_t XECTOOLS::IsInShadow(const TVector3 src, const TVector3 dest)
{
   // check if two points can be directly connected by a linear function
   // in the calorimeter.
   // parameters have to be initialized by InitXECGeometryParameters before calling this.

   // Two points are on a line parallel to the z axis.
   if (src(0) == dest(0) && src(1) == dest(1)) {
      return kFALSE;
   }

   // "src" or "dest" is inner than inner face.
   if (src(0) * src(0) + src(1) * src(1) < (gXERIN * gXERIN) ||
       dest(0) * dest(0) + dest(1) * dest(1) < (gXERIN * gXERIN)) {
      return kTRUE;
   }

   // no intersection with inner face
   if ((dest(0) * src(1) - dest(1) * src(0)) * (dest(0) * src(1) - dest(1) * src(0)) /
       ((dest(0) - src(0)) * (dest(0) - src(0)) + (dest(1) - src(1)) * (dest(1) - src(1))) >= (gXERIN * gXERIN)) {
      return kFALSE;
   }

   // check if closest point to the origine from the line is between "src" and "dest"
   Double_t a = (src(0) * (src(0) - dest(0)) + src(1) * (src(1) - dest(1))) /
                ((dest(0) - src(0)) * (dest(0) - src(0)) + (dest(1) - src(1)) * (dest(1) - src(1)));

   return (a > 0 && a < 1);
}

//______________________________________________________________________________
void XECTOOLS::GammaRayCrossingUVW(const Double_t *vertex, Double_t costh, Double_t phi, Double_t r,
                                   Double_t *uvw)
{
   // Return U,V,W of crossing point with a line from "vertex" with costh/phi direction and radius="r".
   // "r" is radius in cylindrical polar coordinates.
   // "phi" in radian.

   TVector3 vertvec(vertex);

   TVector3 dirvec(1, 0, 0);
   dirvec.SetTheta(TMath::ACos(costh));
   dirvec.SetPhi(phi);

   Double_t dir[2] = {dirvec.X(), dirvec.Y()};

   Double_t w2 = -1 * (dir[0] * vertex[1] - dir[1] * vertex[0]) * (dir[0] * vertex[1] - dir[1] * vertex[0]) +
                 r * r * (dir[0] * dir[0] + dir[1] * dir[1]);
   if (w2 < 0) {
      // invalid input vectors
      uvw[0] = kXECInvalidU;
      uvw[1] = kXECInvalidV;
      uvw[2] = kXECInvalidW;
      return;
   }

   Double_t pol = 1;
   if (vertvec.Mag2() > r * r) {
      if (vertvec * dirvec > 0) {
         // invalid input vectors
         uvw[0] = kXECInvalidU;
         uvw[1] = kXECInvalidV;
         uvw[2] = kXECInvalidW;
         return;
      }
      pol = -1;
   }

   TVector3 crossvec(vertvec + (-1 * (dir[0] * vertex[0] + dir[1] * vertex[1]) + pol * TMath::Sqrt(w2)) /
                     (dir[0] * dir[0] + dir[1] * dir[1]) * dirvec);

   Double_t crossing[3] = {crossvec.X(), crossvec.Y(), crossvec.Z()};
   XYZ2UVW(crossing, uvw);
}

//______________________________________________________________________________
Double_t XECTOOLS::GetHistCorrection(Int_t method, Double_t x, TH1D *hist, TSpline *sp)
{
   // function to extract interplated or extrapolated correction from histogram.

   Int_t    bin       = hist->FindBin(x);
   Double_t binCenter = hist->GetBinCenter(bin);
   Int_t    nextBin   = x > binCenter ? bin + 1 : bin - 1;
   Double_t nextBinCenter;
   Bool_t   spArg     = sp;

   Double_t correction;

   switch (method) {
   case 0:
      // none
      if (x <= hist->GetBinCenter(1)) {
         correction = hist->GetBinContent(1);
      } else if (x >= hist->GetBinCenter(hist->GetNbinsX())) {
         correction = hist->GetBinContent(hist->GetNbinsX());
      } else {
         correction = hist->GetBinContent(bin);
      }
      break;
   case 1:
   default:
      // liner
      if (nextBin <= 1) {
         correction = hist->GetBinContent(1);
      } else if (nextBin >= hist->GetNbinsX()) {
         correction = hist->GetBinContent(hist->GetNbinsX());
      } else {
         // interpolate
         nextBinCenter = hist->GetBinCenter(nextBin);
         correction =
            hist->GetBinContent(bin) * TMath::Abs((nextBinCenter - x) / (nextBinCenter - binCenter)) +
            hist->GetBinContent(nextBin) * TMath::Abs((binCenter - x) / (nextBinCenter - binCenter));
      }
      break;
   case 2:
      // third spline
      if (x <= hist->GetBinCenter(1)) {
         correction = hist->GetBinContent(1);
      } else if (x >= hist->GetBinCenter(hist->GetNbinsX())) {
         correction = hist->GetBinContent(hist->GetNbinsX());
      } else {
         if (!spArg) {
            sp = new TSpline3("sp", hist->GetBinCenter(1), hist->GetBinCenter(hist->GetNbinsX()),
                              hist->GetArray() + 1, hist->GetNbinsX());
         }
         correction = sp->Eval(x);
         if (!spArg) {
            delete sp;
         }
      }
      break;
   case 3:
      // third spline, various bins
      if (x <= hist->GetBinCenter(1)) {
         correction = hist->GetBinContent(1);
      } else if (x >= hist->GetBinCenter(hist->GetNbinsX())) {
         correction = hist->GetBinContent(hist->GetNbinsX());
      } else {
         TGraph *spgr = new TGraph(hist);
         correction = spgr->Eval(x, 0, "S");
         delete spgr;
      }
      break;
   }

   return correction;
}

//______________________________________________________________________________
Double_t XECTOOLS::GetHistCorrection2D(Int_t method, Double_t x, TH1D **histx, Int_t nx,
                                       Double_t y, TH1D **histy, Int_t ny)
{
   // function to extract interpolated or extrapolated correction from histogram.
   // using 1 histogram along y and nx histograms along x

   Int_t    binx        = TMath::Min(histx[0]->GetNbinsX(), TMath::Max(1, histx[0]->FindBin(x)));
   Double_t binCenterX  = histx[0]->GetBinCenter(binx);
   Int_t    nextBinX    = TMath::Min(histx[0]->GetNbinsX(), TMath::Max(1, x > binCenterX ? binx + 1 : binx - 1));

   Int_t    biny        = TMath::Min(histy[0]->GetNbinsX(), TMath::Max(1, histy[0]->FindBin(y)));

   Double_t dx          = (histy[0]->GetXaxis()->GetXmax() - histy[0]->GetXaxis()->GetXmin()) / nx;
   Int_t    ix          = TMath::Min(nx - 1, TMath::Max(0,
                                                        static_cast<Int_t>((y - histy[0]->GetXaxis()->GetXmin()) / dx)));
   Int_t    nextix      = TMath::Min(nx - 1, TMath::Max(0,
                                                        y > histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5) ? ix + 1 : ix - 1));

   Int_t    histBinY             = histy[0]->FindBin(histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5));
   Int_t    histNextBinY         = histy[0]->FindBin(histy[0]->GetXaxis()->GetXmin() + dx * (nextix + 0.5));
   Double_t histCenterY          = histy[0]->GetBinCenter(histBinY);
   Double_t histNextCenterY      = histy[0]->GetBinCenter(histNextBinY);

   Int_t    nextHistBinY         = TMath::Min(histy[0]->GetNbinsX(),
                                              TMath::Max(1, histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5) > histCenterY ?
                                                         histBinY + 1 : histBinY - 1));
   Int_t    nextHistNextBinY     = TMath::Min(histy[0]->GetNbinsX(),
                                              TMath::Max(1, histy[0]->GetXaxis()->GetXmin() + dx * (nextix + 0.5) > histNextCenterY ?
                                                         histNextBinY + 1 : histNextBinY - 1));
   Double_t nextHistCenterY      = histy[0]->GetBinCenter(nextHistBinY);
   Double_t nextHistNextCenterY  = histy[0]->GetBinCenter(nextHistNextBinY);

   Double_t correction = 0;
   Bool_t addDirectoryStatusOrg;

   switch (method) {
   case 0:
      // none
      if (y <= histy[0]->GetBinCenter(1)) {
         correction = histy[0]->GetBinContent(1);
      } else if (y >= histy[0]->GetBinCenter(histy[0]->GetNbinsX())) {
         correction = histy[0]->GetBinContent(histy[0]->GetNbinsX());
      } else {
         correction = histy[0]->GetBinContent(biny);
      }
      if (x <= histx[ix]->GetBinCenter(1)) {
         correction *= histx[ix]->GetBinContent(1);
      } else if (x >= histx[ix]->GetBinCenter(histx[ix]->GetNbinsX())) {
         correction *= histx[ix]->GetBinContent(histx[ix]->GetNbinsX());
      } else {
         correction *= histx[ix]->GetBinContent(binx);
      }
      break;
   case 1:
   default:
      // liner
      Double_t vref[4];

      Double_t vtmp[2];
      vtmp[0] = histy[0]->GetBinContent(histBinY);
      vtmp[1] = histy[0]->GetBinContent(nextHistBinY);
      vref[0] = histx[ix]->GetBinContent(binx) *
                LinearInterpolate(histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5),      vtmp, histCenterY, nextHistCenterY);
      vref[1] =  histx[ix]->GetBinContent(nextBinX) *
                 LinearInterpolate(histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5),      vtmp, histCenterY, nextHistCenterY);
      vtmp[0] = histy[0]->GetBinContent(histNextBinY);
      vtmp[1] = histy[0]->GetBinContent(nextHistNextBinY);
      vref[2] = histx[nextix]->GetBinContent(binx) *
                LinearInterpolate(histy[0]->GetXaxis()->GetXmin() + dx * (nextix + 0.5),  vtmp, histNextCenterY,
                                  nextHistNextCenterY);
      vref[3] = histx[nextix]->GetBinContent(nextBinX) *
                LinearInterpolate(histy[0]->GetXaxis()->GetXmin() + dx * (nextix + 0.5),  vtmp, histNextCenterY,
                                  nextHistNextCenterY);
      correction = LinearInterpolate2D(x, y, vref,
                                       histx[0]->GetBinCenter(binx), histy[0]->GetXaxis()->GetXmin() + dx * (ix + 0.5),
                                       histx[0]->GetBinCenter(nextBinX) - histx[0]->GetBinCenter(binx),
                                       histNextCenterY - histCenterY);
      break;
   case 2:
      // third spline
      // scale by the factor of splineY at input y
      //    splines along x are normalized at each x by 1 spline(use histy[0]) along y, and make new splineY along y at input x
      TSpline3 * spx, *spy;
      if (x <= histx[0]->GetBinCenter(1)) {
         x = histx[0]->GetBinCenter(1) + 1e-10;
      } else if (x >= histx[0]->GetBinCenter(histx[0]->GetNbinsX())) {
         x = histx[0]->GetBinCenter(histx[0]->GetNbinsX()) - 1e-10;
      }
      if (y <= histy[0]->GetBinCenter(1)) {
         spx = new TSpline3("spx", histx[0]->GetBinCenter(1), histx[0]->GetBinCenter(histx[0]->GetNbinsX()),
                            histx[0]->GetArray() + 1, histx[0]->GetNbinsX());
         correction = histy[0]->GetBinContent(1) * spx->Eval(x);
         delete spx;
      } else if (y >= histy[0]->GetBinCenter(histy[0]->GetNbinsX())) {
         spx = new TSpline3("spx", histx[nx - 1]->GetBinCenter(1),
                            histx[nx - 1]->GetBinCenter(histx[nx - 1]->GetNbinsX()), histx[nx - 1]->GetArray() + 1,
                            histx[nx - 1]->GetNbinsX());
         correction = histy[0]->GetBinContent(histy[0]->GetNbinsX()) * spx->Eval(x);
         delete spx;
      } else {
         addDirectoryStatusOrg  = TH1::AddDirectoryStatus();
         TH1::AddDirectory(kFALSE);
         TH1D * hisytmp = new TH1D("histoytmp", "histoytmp", histy[0]->GetNbinsX(), histy[0]->GetXaxis()->GetXmin(),
                                   histy[0]->GetXaxis()->GetXmax());
         hisytmp->ResetStats();
         TH1::AddDirectory(addDirectoryStatusOrg);
         for (Int_t i = 0; i < ny; i++) {
            hisytmp->Add(histy[i], 1 / static_cast<Double_t>(ny));
         }
         spy = new TSpline3("spyonnormalized", histy[0]->GetBinCenter(1),
                            histy[0]->GetBinCenter(histy[0]->GetNbinsX()),
                            hisytmp->GetArray() + 1, histy[0]->GetNbinsX());
         Double_t *xfactor = new Double_t[nx];
         for (Int_t i = 0; i < nx; i++) {
            spx = new TSpline3("spxondify", histx[i]->GetBinCenter(1), histx[i]->GetBinCenter(histx[i]->GetNbinsX()),
                               histx[i]->GetArray() + 1, histx[i]->GetNbinsX());
            xfactor[i] = spy->Eval(dx * (i + 0.5 + 1e-10) + histy[0]->GetXaxis()->GetXmin()) * spx->Eval(x);
            if (nx == 1) {
               correction = spy->Eval(y) * spx->Eval(x);
            }
            delete spx;
         }
         delete spy;
         if (nx != 1) {
            spy = new TSpline3("spyonnormalized", histy[0]->GetBinCenter(1),
                               histy[0]->GetBinCenter(histy[0]->GetNbinsX()),
                               xfactor, nx);
            correction = spy->Eval(y);
            delete spy;
         }
         delete [] xfactor;
         delete hisytmp;
      }
      break;
   }

   return correction;
}

//______________________________________________________________________________
Double_t XECTOOLS::GetHistCorrection2D(Int_t method, Double_t x, Double_t y, TH2D *hist)
{
   // function to extract interpolated or extrapolated correction from histogram.

   const Int_t    binx           = TMath::Min(hist->GetNbinsX(), TMath::Max(1, hist->GetXaxis()->FindBin(x)));
   const Double_t binCenterX     = hist->GetXaxis()->GetBinCenter(binx);
   const Int_t    nextBinX       = TMath::Min(hist->GetNbinsX(), TMath::Max(1, x > binCenterX ? binx + 1 : binx - 1));
   const Double_t nextBinCenterX = hist->GetXaxis()->GetBinCenter(nextBinX);

   const Int_t    biny           = TMath::Min(hist->GetNbinsY(), TMath::Max(1, hist->GetYaxis()->FindBin(y)));
   const Double_t binCenterY     = hist->GetYaxis()->GetBinCenter(biny);
   const Int_t    nextBinY       = TMath::Min(hist->GetNbinsY(), TMath::Max(1, y > binCenterY ? biny + 1 : biny - 1));
   const Double_t nextBinCenterY = hist->GetYaxis()->GetBinCenter(nextBinY);

   Double_t correction = 0;

   switch (method) {
   case 0:
      // raw
      correction = hist->GetBinContent(
                      (x <= hist->GetXaxis()->GetBinCenter(1)) ? 1 :
                      (x >= hist->GetXaxis()->GetBinCenter(hist->GetNbinsX()) ? hist->GetNbinsX() : binx),
                      (y <= hist->GetYaxis()->GetBinCenter(1)) ? 1 :
                      (y >= hist->GetYaxis()->GetBinCenter(hist->GetNbinsY()) ? hist->GetNbinsY() : biny));
      break;
   case 1:
   default:
      // linear
      Double_t vref[2];
      if (x < hist->GetXaxis()->GetXmin()) {
         if (y < hist->GetYaxis()->GetXmin()) {
            correction = hist->GetBinContent(1, 1);
         } else if (y > hist->GetYaxis()->GetXmax()) {
            correction = hist->GetBinContent(1, hist->GetNbinsY());
         } else {
            vref[0] = hist->GetBinContent(1, biny);
            vref[1] = hist->GetBinContent(1, nextBinY);
            correction = LinearInterpolate(y, vref, binCenterY, nextBinCenterY);
         }
      } else if (x > hist->GetXaxis()->GetXmax()) {
         if (y < hist->GetYaxis()->GetXmin()) {
            correction = hist->GetBinContent(hist->GetNbinsX(), 1);
         } else if (y > hist->GetYaxis()->GetXmax()) {
            correction = hist->GetBinContent(hist->GetNbinsX(), hist->GetNbinsY());
         } else {
            vref[0] = hist->GetBinContent(hist->GetNbinsX(), biny);
            vref[1] = hist->GetBinContent(hist->GetNbinsX(), nextBinY);
            correction = LinearInterpolate(y, vref, binCenterY, nextBinCenterY);
         }
      } else {
         if (y < hist->GetYaxis()->GetXmin()) {
            vref[0] = hist->GetBinContent(binx, 1);
            vref[1] = hist->GetBinContent(nextBinX, 1);
            correction = LinearInterpolate(x, vref, binCenterX, nextBinCenterX);
         } else if (y > hist->GetYaxis()->GetXmax()) {
            vref[0] = hist->GetBinContent(binx, hist->GetNbinsY());
            vref[1] = hist->GetBinContent(nextBinX, hist->GetNbinsY());
            correction = LinearInterpolate(x, vref, binCenterX, nextBinCenterX);
         } else {
            // (x, y) position is inside histogram domain.
            // Bilinear interpolation defined in ROOT.
            // https://root.cern.ch/doc/master/TH2_8cxx_source.html#l01366
            correction = hist->Interpolate(x, y);
         }
      }
      break;
   case 2:
      // third spline
      if (x <= hist->GetXaxis()->GetBinCenter(1)) {
         x = hist->GetXaxis()->GetBinCenter(1) + 1e-10;
      } else if (x >= hist->GetXaxis()->GetBinCenter(hist->GetNbinsX())) {
         x = hist->GetXaxis()->GetBinCenter(hist->GetNbinsX()) - 1e-10;
      }
      if (y <= hist->GetYaxis()->GetBinCenter(1)) {
         y = hist->GetYaxis()->GetBinCenter(1) + 1e-10;
      } else if (y >= hist->GetYaxis()->GetBinCenter(hist->GetNbinsY())) {
         y = hist->GetYaxis()->GetBinCenter(hist->GetNbinsY()) - 1e-10;
      } else {
         TSpline3 *spx = 0;
         TSpline3 *spy = 0;
         Double_t *xfactor = new Double_t[hist->GetNbinsX()];
         Double_t *yfactor = new Double_t[hist->GetNbinsY()];
         for (Int_t i = 0; i < hist->GetNbinsY() ; i++) {
            for (Int_t j = 0; j < hist->GetNbinsX(); j++) {
               xfactor[j] = hist->GetBinContent(j + 1, i + 1);
            }
            spx = new TSpline3("spx", hist->GetXaxis()->GetBinCenter(1),
                               hist->GetXaxis()->GetBinCenter(hist->GetNbinsX()),
                               xfactor, hist->GetNbinsX());
            yfactor[i] = spx->Eval(x);
            delete spx;
         }
         if (hist->GetNbinsY() == 1) {
            correction = yfactor[0];
         }
         if (hist->GetNbinsY() != 1) {
            spy = new TSpline3("spy", hist->GetYaxis()->GetBinCenter(1),
                               hist->GetYaxis()->GetBinCenter(hist->GetNbinsY()),
                               yfactor, hist->GetNbinsY());
            correction = spy->Eval(y);
            delete spy;
         }
         delete [] xfactor;
         delete [] yfactor;
      }
      break;
   }

   return correction;
}

//______________________________________________________________________________
Double_t XECTOOLS::GetHistCorrection3D(Int_t method, Double_t x, Double_t y, Double_t z, TH3D *hist)
{
   // method
   // 1st digit:  Z 1D hist correction method
   // 2nd digit: XY 2D hist correction method
   Int_t method_z = method % 10;
   Int_t bin = hist->GetXaxis()->FindBin(x);
   bin = std::max(bin, 1);
   bin = std::min(bin, hist->GetNbinsX());
   hist->GetXaxis()->SetRange(bin, bin);

   bin = hist->GetYaxis()->FindBin(y);
   bin = std::max(bin, 1);
   bin = std::min(bin, hist->GetNbinsY());
   hist->GetYaxis()->SetRange(bin, bin);

   TH1D* hist_pz = (TH1D*)hist->Project3D("z");
   Double_t correction_z = GetHistCorrection(method_z, z, hist_pz);
   Double_t normalization_xy = GetHistCorrection(0, z, hist_pz);

   hist->GetXaxis()->SetRange();
   hist->GetYaxis()->SetRange();

   Int_t method_xy = method / 10;
   Double_t correction_xy;
   switch (method_xy) {
   case 0:
   default:
      correction_xy = 1;
      break;
   case 1:
   case 2:
      bin = hist->GetZaxis()->FindBin(z);
      bin = std::max(bin, 1);
      bin = std::min(bin, hist->GetNbinsZ());
      hist->GetZaxis()->SetRange(bin, bin);

      TH2D* hist_pxy = (TH2D*)hist->Project3D("yx");
      if (y <= hist_pxy->GetYaxis()->GetBinCenter(1)) {
         y = hist_pxy->GetYaxis()->GetBinCenter(1) + 1e-10;
      } else if (y >= hist_pxy->GetYaxis()->GetBinCenter(hist_pxy->GetNbinsY())) {
         y = hist_pxy->GetYaxis()->GetBinCenter(hist_pxy->GetNbinsY()) - 1e-10;
      }
      correction_xy = GetHistCorrection2D(method_xy, x, y, hist_pxy) / normalization_xy;
      break;
   }
   hist->GetZaxis()->SetRange();
   return correction_z * correction_xy;
}

//______________________________________________________________________________
Double_t XECTOOLS::LinearInterpolate(Double_t x, Double_t *vref, Double_t x1, Double_t x2)
{
   if (!vref) {
      return 0;
   }
   if (x1 == x2) {
      return TMath::Mean(2, vref);
   }
   Double_t xmin;
   Double_t xmax;
   Int_t    imin;
   Int_t    imax;
   if (x2 > x1) {
      xmin = x1;
      xmax = x2;
      imin = 0;
      imax = 1;
   } else {
      xmin = x2;
      xmax = x1;
      imin = 1;
      imax = 0;
   }
   if (x >= xmax) {
      return vref[imax];
   } else if (x <= xmin) {
      return vref[imin];
   }
   return (vref[imin] * (xmax - x) + vref[imax] * (x - xmin)) / (xmax - xmin);
}

//______________________________________________________________________________
Double_t XECTOOLS::LinearInterpolate2D(Double_t x, Double_t y, Double_t *vref,
                                       Double_t x0, Double_t y0, Double_t dx, Double_t dy)
{
   // vref[2] = v(x0, y0 + dy)  vref[3] = v(x0 + dx, y0 + dy)
   // vref[0] = v(x0, y0)       vref[1] = v(x0 + dx, y0)
   Double_t vtmp[2];

   if (!vref) {
      return 0;
   }

   if (dx == 0 && dy == 0) {
      return TMath::Mean(4, vref);
   } else if (dx == 0) {
      vtmp[0] = (vref[0] + vref[1]) * 0.5;
      vtmp[1] = (vref[2] + vref[3]) * 0.5;
      return LinearInterpolate(y, vtmp, y0, y0 + dy);
   } else if (dy == 0) {
      vtmp[0] = (vref[0] + vref[2]) * 0.5;
      vtmp[1] = (vref[1] + vref[3]) * 0.5;
      return LinearInterpolate(x, vtmp, x0, x0 + dx);
   }

   Double_t xlb, ylb, vlb;
   Double_t xrb, yrb, vrb;
   Double_t xlt, ylt, vlt;
   Double_t xrt, yrt, vrt;

   if (dx > 0) {
      if (dy > 0) {
         xlb = x0;
         ylb = y0;
         vlb = vref[0];
         xrb = x0 + dx;
         yrb = y0;
         vrb = vref[1];
         xlt = x0;
         ylt = y0 + dy;
         vlt = vref[2];
         xrt = x0 + dx;
         yrt = y0 + dy;
         vrt = vref[3];
      } else {
         xlt = x0;
         ylt = y0;
         vlt = vref[0];
         xrt = x0 + dx;
         yrt = y0;
         vrt = vref[1];
         xlb = x0;
         ylb = y0 + dy;
         vlb = vref[2];
         xrb = x0 + dx;
         yrb = y0 + dy;
         vrb = vref[3];
      }
   } else {
      if (dy > 0) {
         xrb = x0;
         yrb = y0;
         vrb = vref[0];
         xlb = x0 + dx;
         ylb = y0;
         vlb = vref[1];
         xrt = x0;
         yrt = y0 + dy;
         vrt = vref[2];
         xlt = x0 + dx;
         ylt = y0 + dy;
         vlt = vref[3];
      } else {
         xrt = x0;
         yrt = y0;
         vrt = vref[0];
         xlt = x0 + dx;
         ylt = y0;
         vlt = vref[1];
         xrb = x0;
         yrb = y0 + dy;
         vrb = vref[2];
         xlb = x0 + dx;
         ylb = y0 + dy;
         vlb = vref[3];
      }
   }

   if (x >= xrt) {
      if (y >= yrt) {
         return vrt;
      } else if (y <= yrb) {
         return vrb;
      } else {
         vtmp[0] = vrb;
         vtmp[1] = vrt;
         return LinearInterpolate(y, vtmp, yrb, yrt);
      }
   } else if (x <= xlt) {
      if (y >= ylt) {
         return vlt;
      } else if (y <= ylb) {
         return vlb;
      } else {
         vtmp[0] = vlb;
         vtmp[1] = vlt;
         return LinearInterpolate(y, vtmp, ylb, ylt);
      }
   } else if (y >= ylt) {
      vtmp[0] = vlt;
      vtmp[1] = vrt;
      return LinearInterpolate(x, vtmp, xlt, xrt);
   } else if (y <= ylb) {
      vtmp[0] = vlb;
      vtmp[1] = vrb;
      return LinearInterpolate(x, vtmp, xlb, xrb);
   }

   Double_t xc = x0 + dx / 2;
   Double_t yc = y0 + dy / 2;
   Double_t vc = TMath::Mean(4, vref);

   Double_t xa, ya, va;
   Double_t xb, yb, vb;

   dx = TMath::Abs(dx);
   dy = TMath::Abs(dy);

   if (y >=  dy * x / dx + ylb) {
      xa = xlt;
      ya = ylt;
      va = vlt;
      if (y >= -dy * x / dx + ylt) {
         xb = xrt;
         yb = yrt;
         vb = vrt;
      } else {
         xb = xlb;
         yb = ylb;
         vb = vlb;
      }
   } else {
      xa = xrb;
      ya = yrb;
      va = vrb;
      if (y >= -dy * x / dx + ylt) {
         xb = xrt;
         yb = yrt;
         vb = vrt;
      } else {
         xb = xlb;
         yb = ylb;
         vb = vlb;
      }
   }

   Double_t A =
      ((x  - xc) * (yb - yc) - (xb - xc) * (y  - yc)) /
      ((xa - xc) * (yb - yc) - (xb - xc) * (ya - yc));
   Double_t B =
      ((x  - xc) * (ya - yc) - (xa - xc) * (y  - yc)) /
      ((xb - xc) * (ya - yc) - (xa - xc) * (yb - yc));

   return (va - vc) * A + (vb - vc) * B + vc;
}


// Solid angle stuff (Giovanni Signorelli, 20.04.2009)
#define MY_PI   3.14159265359
#define MEG_DBL_EPSILON        2.2204460492503131e-16
#define MEG_SQRT_DBL_EPSILON   1.4901161193847656e-08
#define MEG_DBL_MIN        2.2250738585072014e-308
#define MEG_DBL_MAX        1.7976931348623157e+308
#define MEG_PREC_DOUBLE  0
#define MEG_MAX(a,b) ((a) > (b) ? (a) : (b))
#define MEG_MIN(a,b) ((a) < (b) ? (a) : (b))

Double_t locMAX3(Double_t x, Double_t y, Double_t z)
{
   Double_t xy = MEG_MAX(x, y);
   return MEG_MAX(xy, z);
}

Double_t locMAX4(Double_t x, Double_t y, Double_t z, Double_t w)
{
   Double_t xy  = MEG_MAX(x,  y);
   Double_t xyz = MEG_MAX(xy, z);
   return MEG_MAX(xyz, w);
}

Double_t meg_sf_ellint_RF_e(Double_t x, Double_t y, Double_t z)
{
   const Double_t lolim = 5.0 * MEG_DBL_MIN;
   const Double_t uplim = 0.2 * MEG_DBL_MAX;
   const Double_t errtol = 0.001;

   if (x < 0.0 || y < 0.0 || z < 0.0) {
      return -1; //DOMAIN_ERROR(result);
   } else if (x + y < lolim || x + z < lolim || y + z < lolim) {
      return -1; //DOMAIN_ERROR(result);
   } else if (locMAX3(x, y, z) < uplim) {
      const Double_t c1 = 1.0 / 24.0;
      const Double_t c2 = 3.0 / 44.0;
      const Double_t c3 = 1.0 / 14.0;
      Double_t xn = x;
      Double_t yn = y;
      Double_t zn = z;
      Double_t mu, xndev, yndev, zndev, e2, e3, s;
      while (1) {
         Double_t epslon, lamda;
         Double_t xnroot, ynroot, znroot;
         mu = (xn + yn + zn) / 3.0;
         xndev = 2.0 - (mu + xn) / mu;
         yndev = 2.0 - (mu + yn) / mu;
         zndev = 2.0 - (mu + zn) / mu;
         epslon = locMAX3(fabs(xndev), fabs(yndev), fabs(zndev));
         if (epslon < errtol) {
            break;
         }
         xnroot = sqrt(xn);
         ynroot = sqrt(yn);
         znroot = sqrt(zn);
         lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
         xn = (xn + lamda) * 0.25;
         yn = (yn + lamda) * 0.25;
         zn = (zn + lamda) * 0.25;
      }
      e2 = xndev * yndev - zndev * zndev;
      e3 = xndev * yndev * zndev;
      s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3;
      return s / sqrt(mu);
   } else {
      return -1; //DOMAIN_ERROR(result);
   }
}

Double_t meg_sf_ellint_RC_e(Double_t x, Double_t y)
{
   const Double_t lolim = 5.0 * MEG_DBL_MIN;
   const Double_t uplim = 0.2 * MEG_DBL_MAX;
   const Double_t errtol = 0.001;

   if (x < 0.0 || y < 0.0 || x + y < lolim) {
      return -1; //DOMAIN_ERROR(result);
   } else if (MEG_MAX(x, y) < uplim) {
      const Double_t c1 = 1.0 / 7.0;
      const Double_t c2 = 9.0 / 22.0;
      Double_t xn = x;
      Double_t yn = y;
      Double_t mu, sn, lamda, s;
      while (1) {
         mu = (xn + yn + yn) / 3.0;
         sn = (yn + mu) / mu - 2.0;
         if (fabs(sn) < errtol) {
            break;
         }
         lamda = 2.0 * sqrt(xn) * sqrt(yn) + yn;
         xn = (xn + lamda) * 0.25;
         yn = (yn + lamda) * 0.25;
      }
      s = sn * sn * (0.3 + sn * (c1 + sn * (0.375 + sn * c2)));
      return (1.0 + s) / sqrt(mu);
   } else {
      return -1; //DOMAIN_ERROR(result);
   }
}

Double_t meg_sf_ellint_RD_e(Double_t x, Double_t y, Double_t z)
{
   const Double_t errtol = 0.001;
   const Double_t lolim = 2.0 / pow(MEG_DBL_MAX, 2.0 / 3.0);
   const Double_t uplim = pow(0.1 * errtol / MEG_DBL_MIN, 2.0 / 3.0);

   if (MEG_MIN(x, y) < 0.0 || MEG_MIN(x + y, z) < lolim) {
      return -1; //DOMAIN_ERROR(result);
   } else if (locMAX3(x, y, z) < uplim) {
      const Double_t c1 = 3.0 / 14.0;
      const Double_t c2 = 1.0 /  6.0;
      const Double_t c3 = 9.0 / 22.0;
      const Double_t c4 = 3.0 / 26.0;
      Double_t xn = x;
      Double_t yn = y;
      Double_t zn = z;
      Double_t sigma  = 0.0;
      Double_t power4 = 1.0;
      Double_t ea, eb, ec, ed, ef, s1, s2;
      Double_t mu, xndev, yndev, zndev;
      while (1) {
         Double_t xnroot, ynroot, znroot, lamda;
         Double_t epslon;
         mu = (xn + yn + 3.0 * zn) * 0.2;
         xndev = (mu - xn) / mu;
         yndev = (mu - yn) / mu;
         zndev = (mu - zn) / mu;
         epslon = locMAX3(fabs(xndev), fabs(yndev), fabs(zndev));
         if (epslon < errtol) {
            break;
         }
         xnroot = sqrt(xn);
         ynroot = sqrt(yn);
         znroot = sqrt(zn);
         lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
         sigma  += power4 / (znroot * (zn + lamda));
         power4 *= 0.25;
         xn = (xn + lamda) * 0.25;
         yn = (yn + lamda) * 0.25;
         zn = (zn + lamda) * 0.25;
      }
      ea = xndev * yndev;
      eb = zndev * zndev;
      ec = ea - eb;
      ed = ea - 6.0 * eb;
      ef = ed + ec + ec;
      s1 = ed * (- c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef);
      s2 = zndev * (c2 * ef + zndev * (- c3 * ec + zndev * c4 * ea));
      return 3.0 * sigma + power4 * (1.0 + s1 + s2) / (mu * sqrt(mu));
   } else {
      return -1;// DOMAIN_ERROR(result);
   }
}

Double_t meg_sf_ellint_RJ_e(Double_t x, Double_t y, Double_t z, Double_t p)
{
   const Double_t errtol = 0.001;
   const Double_t lolim =       pow(5.0 * MEG_DBL_MIN, 1.0 / 3.0);
   const Double_t uplim = 0.3 * pow(0.2 * MEG_DBL_MAX, 1.0 / 3.0);

   if (x < 0.0 || y < 0.0 || z < 0.0) {
      return -1; // DOMAIN_ERROR(result);
   } else if (x + y < lolim || x + z < lolim || y + z < lolim || p < lolim) {
      return -1; //DOMAIN_ERROR(result);
   } else if (locMAX4(x, y, z, p) < uplim) {
      const Double_t c1 = 3.0 / 14.0;
      const Double_t c2 = 1.0 /  3.0;
      const Double_t c3 = 3.0 / 22.0;
      const Double_t c4 = 3.0 / 26.0;
      Double_t xn = x;
      Double_t yn = y;
      Double_t zn = z;
      Double_t pn = p;
      Double_t sigma = 0.0;
      Double_t power4 = 1.0;
      Double_t mu, xndev, yndev, zndev, pndev;
      Double_t ea, eb, ec, e2, e3, s1, s2, s3;
      while (1) {
         Double_t xnroot, ynroot, znroot;
         Double_t lamda, alfa, beta;
         Double_t epslon;
         mu = (xn + yn + zn + pn + pn) * 0.2;
         xndev = (mu - xn) / mu;
         yndev = (mu - yn) / mu;
         zndev = (mu - zn) / mu;
         pndev = (mu - pn) / mu;
         epslon = locMAX4(fabs(xndev), fabs(yndev), fabs(zndev), fabs(pndev));
         if (epslon < errtol) {
            break;
         }
         xnroot = sqrt(xn);
         ynroot = sqrt(yn);
         znroot = sqrt(zn);
         lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
         alfa = pn * (xnroot + ynroot + znroot) + xnroot * ynroot * znroot;
         alfa = alfa * alfa;
         beta = pn * (pn + lamda) * (pn + lamda);
         Double_t rc;
         rc = meg_sf_ellint_RC_e(alfa, beta);
         sigma  += power4 * rc;
         power4 *= 0.25;
         xn = (xn + lamda) * 0.25;
         yn = (yn + lamda) * 0.25;
         zn = (zn + lamda) * 0.25;
         pn = (pn + lamda) * 0.25;
      }
      ea = xndev * (yndev + zndev) + yndev * zndev;
      eb = xndev * yndev * zndev;
      ec = pndev * pndev;
      e2 = ea - 3.0 * ec;
      e3 = eb + 2.0 * pndev * (ea - ec);
      s1 = 1.0 + e2 * (- c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3);
      s2 = eb * (0.5 * c2 + pndev * (- c3 - c3 + pndev * c4));
      s3 = pndev * ea * (c2 - pndev * c3) - c2 * pndev * ec;
      return 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * sqrt(mu));
   } else {
      return -1; //DOMAIN_ERROR(result);
   }
}


Double_t comp_ellint_1(Double_t k)
{
   if (k * k >= 1.0) {
      return -1; //DOMAIN_ERROR(result);
   } else if (k * k >= 1.0 - MEG_SQRT_DBL_EPSILON) {
      /* [Abramowitz+Stegun, 17.3.34] */
      const Double_t y = 1.0 - k * k;
      const Double_t a[] = { 1.38629436112, 0.09666344259, 0.03590092383 };
      const Double_t b[] = { 0.5, 0.12498593597, 0.06880248576 };
      const Double_t ta = a[0] + y * (a[1] + y * a[2]);
      const Double_t tb = -log(y) * (b[0] + y * (b[1] + y * b[2]));
      return ta + tb;
   } else {
      /* This was previously computed as,

      return meg_sf_ellint_RF_e(0.0, 1.0 - k*k, 1.0, mode, result);

      but this underestimated the total error for small k, since the
      argument y=1-k^2 is not exact (there is an absolute error of
      MEG_DBL_EPSILON near y=0 due to cancellation in the subtraction).
      Taking the singular behavior of -log(y) above gives an error
      of 0.5*epsilon/y near y=0. (BJG) */

      Double_t y = 1.0 - k * k;
      return meg_sf_ellint_RF_e(0.0, y, 1.0);
   }
}

/* [Carlson, Numer. Math. 33 (1979) 1, (4.3) phi=pi/2] */
Double_t meg_sf_ellint_Pcomp_e(Double_t k, Double_t n)
{
   if (k * k >= 1.0) {
      return -1; //DOMAIN_ERROR(result);
   }
   /* FIXME: need to handle k ~=~ 1  cancellations */
   else {
      Double_t rf;
      Double_t rj;
      const Double_t y = 1.0 - k * k;
      rf = meg_sf_ellint_RF_e(0.0, y, 1.0);
      rj = meg_sf_ellint_RJ_e(0.0, y, 1.0, 1.0 + n);
      return rf - (n / 3.0) * rj;
   }
}

Double_t comp_ellint_3(Double_t n, Double_t k)
{
   return meg_sf_ellint_Pcomp_e(k, n);
}
