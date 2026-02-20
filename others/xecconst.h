// $Id$
#ifndef XECCONST_H
#define XECCONST_H

#include <Rtypes.h>
#include <TMath.h>
#include "units/MEGSystemOfUnits.h"

namespace XECCONSTANTS{
   // constants
   // const Int_t    kMaxNXePM          = 4722;  // maximum number of PMs in history. Don't decrease it.
   const Int_t    kMaxNXePM          = 4722+38;  // maximum number of PMs in history. Don't decrease it.
   const Int_t    kMaxNXeInnerPM     = 44*93; // maximum number of inner PMs in history. Don't decrease it.
   const Int_t    kMaxNXePMT     = 668; // maximum number of PMTs in history. Don't decrease it.
   const Double_t kLXeRefractiveIndex = 1.61; // refractive index of LXe, value is derived from xecal.cards in gem.

   // LXe scintillation decay time constants
   const Double_t kLXeDecayRecombination  = 45  * MEG::nanosecond;
   const Double_t kLXeDecayLongComponent  = 22  * MEG::nanosecond;
   const Double_t kLXeDecayShortComponent = 4.2 * MEG::nanosecond;
   
   /* LXe scintillation decay ratio for electrons */
   const Double_t kLXeDecayRatioRecombinationElectron  = 1;
   const Double_t kLXeDecayRatioLongComponentElectron  = 0;
   const Double_t kLXeDecayRatioShortComponentElectron = 0;
   /* and alpha particles*/
   const Double_t kLXeDecayRatioRecombinationAlpha  = 0;
   const Double_t kLXeDecayRatioLongComponentAlpha  = 1;
   const Double_t kLXeDecayRatioShortComponentAlpha = 0.43;

   const Int_t kInnerFace      = 0;
   const Int_t kOuterFace      = 1;
   const Int_t kUpstreamFace   = 2;
   const Int_t kDownstreamFace = 3;
   const Int_t kTopFace        = 4;
   const Int_t kBottomFace     = 5;
   const Int_t kNumberOfFaces  = 6;

   /*
   NUMPM for top/btm does not reflect modified layout.
   */
   const Int_t kNUMPM1[kNumberOfFaces] = {44, 9, 6, 6, 6, 6};
   const Int_t kNUMPM2[kNumberOfFaces] = {93, 24, 24, 24, 9, 9};

   const Double_t kRCATH = 4.5 / 2. * MEG::centimeter;      // PMT photo-cathode radius
   const Int_t    kNumberOfSiPMPixels[2] = {119*2,117*2};
   const Int_t    kNumberOfGEMSiPMPixels[2] = {118*2,118*2};
}


// namespace XECWAVEFORM{
//    const Double_t kTimeDifFromCF = 12.6 * MEG::nanosecond;
// }

#endif                          //XECCONST_H
