// $Id$
// Calculate Solid angle of PM from a view point
#include <TVector3.h>
#include <TRandom.h>
#include <TMath.h>
#include "generated/MEGXECPMRunHeader.h"
#include "xec/PMSolidAngle.h"
#include "constants/xec/xecconst.h"
#include "units/MEGSystemOfUnits.h"
using namespace MEG;

namespace
{
//______________________________________________________________________________
inline Double_t CalculateReflectivity_P(std::complex<Double_t> n1, std::complex<Double_t> n2, Double_t theta)
{
   //Calculate reflectivity for P wave based on Fresnel equation
   theta *= degree;
   return pow(std::abs(n2 * n2 * cos(theta) - n1 * std::sqrt(n2 * n2 - n1 * n1 * sin(theta) * sin(theta))), 2) /
          pow(std::abs(n2 * n2 * cos(theta) + n1 * std::sqrt(n2 * n2 - n1 * n1 * sin(theta) * sin(theta))), 2);
}

inline Double_t CalculateReflectivity_S(std::complex<Double_t> n1, std::complex<Double_t> n2, Double_t theta)
{
   //Calculate reflectivity for S wave based on Fresnel equation
   theta *= degree;
   return pow(std::abs(n1 * cos(theta) - std::sqrt(n2 * n2 - n1 * n1 * sin(theta) * sin(theta))), 2) /
          pow(std::abs(n1 * cos(theta) + std::sqrt(n2 * n2 - n1 * n1 * sin(theta) * sin(theta))), 2);
}

inline Double_t SphericalTriangle(TVector3& a, TVector3& b, TVector3& c)
{
   // Return solid angle of spherical triangle when viewing from (0, 0, 0).
   // a, b and c are direction vectors. Their length can be any if not zero.
   TVector3 A = a.Cross(b);
   TVector3 B = b.Cross(c);
   TVector3 C = c.Cross(a);
   return A.Angle(-B) + B.Angle(-C) + C.Angle(-A) - TMath::Pi();
}

//______________________________________________________________________________
inline Double_t SphericalHexagon(TVector3& a, TVector3& b, TVector3& c,
                                 TVector3& d, TVector3& e, TVector3& f)
{
   // Return solid angle of spherical hexagon when viewing from (0, 0, 0).
   // a, b and c are direction vectors. Their length can be any if not zero.
   TVector3 A = a.Cross(b);
   TVector3 B = b.Cross(c);
   TVector3 C = c.Cross(d);
   TVector3 D = d.Cross(e);
   TVector3 E = e.Cross(f);
   TVector3 F = f.Cross(a);
   return A.Angle(-B) + B.Angle(-C) + C.Angle(-D)
          + D.Angle(-E) + E.Angle(-F) + F.Angle(-A)
          - 4 * TMath::Pi();
}
} // unnamed namespace

const Double_t kFactor0 = 0.5;
const Double_t kFactor1 = 8.66025403784438597e-01; // sqrt(3)/2

//______________________________________________________________________________

Double_t PMSolidAngle::TransmittanceOfSilicon(Double_t theta)
{
   std::complex<Double_t> n1(1.69, 0); //LXe,quartz
   std::complex<Double_t> n2(0.8, 2.2); //Silicon
   return 1. - CalculateReflectivity_P(n1, n2, theta) / 2.0 - CalculateReflectivity_S(n1, n2, theta) / 2.0;
}

Double_t PMSolidAngle::PMIncidentAngle(TVector3 view, TVector3 center, TVector3 normal)
{
   TVector3 center_view = center - view;
   Double_t incidentAngle = center_view.Angle(normal);
   return incidentAngle;
}

Double_t PMSolidAngle::PMSolidAngleMPPC(TVector3 view, TVector3 center, TVector3 normal)
{
// This function returns solid angle of MPPC.

// check direction.
   TVector3 center_view = center - view;
   if (center_view.Dot(normal) > 0) {
      return 0;
   }

   TVector3 center_chip;
   // Double_t rtnval=0;

   //Set U,V direction. Temporary hard corded.
   TVector3 unit[3]; // unit vectors.
   unit[0].SetXYZ(0, 0, 1); //U direction
   unit[1] = (unit[0].Cross(normal)).Unit();//V direction
   unit[2] = normal.Unit();//W direction
   unit[0] = (unit[1].Cross(unit[2])).Unit();

   //Calculate the Solidangle. Each chip is divided into  (nSegmentation)^2 in the calculation.
   Double_t ChipDistance = 0.5 * millimeter;
   Double_t ChipSize = 5.90 * millimeter;
   Double_t sin_a1;
   Double_t sin_a2;
   Double_t sin_b1;
   Double_t sin_b2;
   Double_t solid_total2 = 0;

   TVector3 vcorner1 = center + ChipDistance / 2.0 * unit[0] + ChipDistance / 2.0 * unit[1];
   TVector3 vcorner2 = vcorner1 + ChipSize * unit[0] + ChipSize * unit[1];
   TVector3 v1       = vcorner1 - view;
   TVector3 v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total2 += TMath::Abs(asin(sin_a1 * sin_b1) +
                              asin(sin_a2 * sin_b2) -
                              asin(sin_a1 * sin_b2) -
                              asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   vcorner1 = center - ChipDistance / 2.0 * unit[0] + ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 - ChipSize * unit[0] + ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total2 += TMath::Abs(asin(sin_a1 * sin_b1) +
                              asin(sin_a2 * sin_b2) -
                              asin(sin_a1 * sin_b2) -
                              asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   vcorner1 = center   + ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 + ChipSize        * unit[0] - ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total2 += TMath::Abs(asin(sin_a1 * sin_b1) +
                              asin(sin_a2 * sin_b2) -
                              asin(sin_a1 * sin_b2) -
                              asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());

   vcorner1 = center   - ChipDistance / 2.0 * unit[0] - ChipDistance / 2.0 * unit[1];
   vcorner2 = vcorner1 - ChipSize        * unit[0] - ChipSize * unit[1];
   v1       = vcorner1 - view;
   v2       = vcorner2 - view;
   sin_a1 = v1.Dot(unit[0]) / sqrt(pow(v1.Dot(unit[0]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_a2 = v2.Dot(unit[0]) / sqrt(pow(v2.Dot(unit[0]), 2) + pow(v2.Dot(unit[2]), 2));
   sin_b1 = v1.Dot(unit[1]) / sqrt(pow(v1.Dot(unit[1]), 2) + pow(v1.Dot(unit[2]), 2));
   sin_b2 = v2.Dot(unit[1]) / sqrt(pow(v2.Dot(unit[1]), 2) + pow(v2.Dot(unit[2]), 2));
   solid_total2 += TMath::Abs(asin(sin_a1 * sin_b1) +
                              asin(sin_a2 * sin_b2) -
                              asin(sin_a1 * sin_b2) -
                              asin(sin_b1 * sin_a2)) / (4 * TMath::Pi());
   // std::cout<<"theory: "<<solid_total2<<std::endl;
   return solid_total2;
}

Double_t PMSolidAngle::PMSolidAnglePMT(TVector3 view, TVector3 center, TVector3 normal)
{
   // Computes solid angle of a point in front of a disc according to the paper of
   // F. Paxton, "Solid Angle Calculation for a Circular Disk", Rev. Sci. Instr. 30 (1959) 254
   //
   // This version returns ZERO if the PM normal is opposite to the required point. Use the other function
   // PMExactSolidAngleWithSign() if the angle is needed also for PMs in shadow.
   //
   // The computation involves complete elliptic integrals of the first and third kind. The reader is referred
   // to the paper for the complete explanation of the symbols, that is just sketched here:
   //
   // P        = The point
   // Rm       = PM radius
   // R0       = Distance on the disc plane from the disk center to the projection of the point P
   // L        = The distance of P from the disc plane
   // Rmax     = Sqrt( L**2 + ( R0 + Rm )**2 )
   // R1       = Sqrt( L**2 + ( R0 - Rm )**2 )
   // kappa    = Sqrt( 1 - R1**2/Rmax**2 )
   // alphasq  = 4 R0 Rm / (R0 + Rm)**2
   // K        = Complete elliptic function of the first kind: comp_ellint_1(k)
   // Pi       = Complete elliptic function of the third kind: comp_ellint_3(alphasquare, k)
   //
   // The result is different for the three cases R0 < Rm, R0 = Rm, R0 > Rm
   //

   TVector3 center_view = view - center;
   if ((center_view) * normal <= 0) {
      return 0;
   }
   // Initializations
   Double_t solid_angle = 0;
   Double_t PiGreco = TMath::Pi();
   //
   // First of all handle special cases:
   if (false) {
      return solid_angle;
   }
   // Radius of photo-kathode
   Double_t Rm      = XECCONSTANTS::kRCATH;
   // Point-to-PM center distance (points FROM the PM TO the point)
   Double_t dist = center_view.Mag();
   // Point-to-disc-plane distance
   Double_t L    = center_view.Dot(normal);
   Double_t Theta = center_view.Angle(normal);
   // Point-projection-to-disc-center distance
   Double_t R0 = dist * TMath::Sin(Theta);
   // Define Rmax
   Double_t Rmax;
   Rmax     = TMath::Sqrt(L * L + (R0 + Rm) * (R0 + Rm));
   // Define R1
   Double_t R1;
   R1       = TMath::Sqrt(L * L + (R0 - Rm) * (R0 - Rm));
   // Define kappa
   Double_t kappa;
   kappa    = TMath::Sqrt(1 - R1 * R1 / Rmax / Rmax);
   // Define alphasq
   Double_t alphasq;
   alphasq  = 4 * R0 * Rm / (R0 + Rm) / (R0 + Rm);
   // Which is my case?
   // std::cout << dist << " " << L / Rm << " " << R0 / Rm << " " << Rm << " " << Rmax << " " << kappa << std::endl;
   if (TMath::Abs((R0 - Rm) / Rm) < 1e-03) {
      solid_angle = PiGreco -
                    2 * L / Rmax * std::comp_ellint_1(kappa);
   } else if (R0 < Rm) {
      solid_angle = 2 * PiGreco -
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

//______________________________________________________________________________

Double_t PMSolidAngle::PMSolidAngleHexagon(TVector3& view, TVector3& center,
                                           TVector3& normal, Double_t R)
{
   // This function returns solid angle of inscribed hexagon in PM cathode.
   // This is not equal to solid angle of PM.

   // check direction.
   TVector3 center_view = center - view;
   if ((center_view) * normal <= 0) {
      return 0;
   }

   TVector3 vec[6];  // vertex of hexagon
   TVector3 unit[2]; // unit vectors. perpendicular to normal
   TVector3 tmp;
   unit[0] = center_view.Cross(normal);
   while (unit[0].Mag2() < 1E-6) { // (center - view) and normal are parallel.
      tmp.SetXYZ(gRandom->Rndm(), gRandom->Rndm(), gRandom->Rndm());
      unit[0] = tmp.Cross(normal);
   }
   unit[0] = unit[0].Unit();
   unit[1] = (unit[0].Cross(normal)).Unit();

   // hexagon
   vec[0] = center_view + (unit[1]) * R;
   vec[1] = center_view + (unit[1] * kFactor0 + unit[0] * kFactor1) * R;
   vec[2] = center_view + (-unit[1] * kFactor0 + unit[0] * kFactor1) * R;
   vec[3] = center_view + (-unit[1]) * R;
   vec[4] = center_view + (-unit[1] * kFactor0 - unit[0] * kFactor1) * R;
   vec[5] = center_view + (unit[1] * kFactor0 - unit[0] * kFactor1) * R;

   return SphericalHexagon(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]);
}

Double_t PMSolidAngle::PMExactSolidAngle(MEGXECPMRunHeader * which_pm, Double_t * point)
{
// Computes solid angle of a point in front of a disc according to the paper of
// F. Paxton, "Solid Angle Calculation for a Circular Disk", Rev. Sci. Instr. 30 (1959) 254
//
// This version returns ZERO if the PM normal is opposite to the required point. Use the other function
// PMExactSolidAngleWithSign() if the angle is needed also for PMs in shadow.
//
// The computation involves complete elliptic integrals of the first and third kind. The reader is referred
// to the paper for the complete explanation of the symbols, that is just sketched here:
//
// P        = The point
// Rm       = PM radius
// R0       = Distance on the disc plane from the disk center to the projection of the point P
// L        = The distance of P from the disc plane
// Rmax     = Sqrt( L**2 + ( R0 + Rm )**2 )
// R1       = Sqrt( L**2 + ( R0 - Rm )**2 )
// kappa    = Sqrt( 1 - R1**2/Rmax**2 )
// alphasq  = 4 R0 Rm / (R0 + Rm)**2
// K        = Complete elliptic function of the first kind: comp_ellint_1(k)
// Pi       = Complete elliptic function of the third kind: comp_ellint_3(alphasquare, k)
//
// The result is different for the three cases R0 < Rm, R0 = Rm, R0 > Rm
//
// Initializations
   Double_t solid_angle = 0;
   Double_t PiGreco = TMath::Pi();
//
// First of all handle special cases:
   if (false) {
      return solid_angle;
   }
// Fetch PM information:
//   -Position:
   Double_t xpm  = which_pm->GetXYZAt(0);
   Double_t ypm  = which_pm->GetXYZAt(1);
   Double_t zpm  = which_pm->GetXYZAt(2);
//   -Normal:
   Double_t xn   = which_pm->GetDirectionAt(0);
   Double_t yn   = which_pm->GetDirectionAt(1);
   Double_t zn   = which_pm->GetDirectionAt(2);
// Fetch POINT information:
   Double_t xP   = point[0];
   Double_t yP   = point[1];
   Double_t zP   = point[2];
// Radius of photo-kathode
   Double_t Rm      = XECCONSTANTS::kRCATH;
// Point-to-PM center distance vector (points FROM the PM TO the point)
   Double_t xd   = xP - xpm;
   Double_t yd   = yP - ypm;
   Double_t zd   = zP - zpm;
   Double_t dist = TMath::Sqrt(xd * xd + yd * yd + zd * zd);
// Point-to-disc-plane distance
   Double_t L;
   L = xn * xd + yn * yd + zn * zd;
// If the point is on the plane => the solid angle is zero.
   if (L == 0) {
      return 0;
   }
// Is this PM in shadow? If L>0 no, if L<0 yes. (remember the sign of d)
// It is defined as an integer to bear the sign of the solid angle.
   Int_t IsInShadow;
   IsInShadow = L > 0 ? +1 : -1;
// Fast version: if the point is behind the PM return -1
   if (IsInShadow < 0) {
      return (Double_t)IsInShadow;
   }
// Point-projection-to-disc-center distance
   Double_t R0;
   R0       = TMath::Sqrt(dist * dist - L * L);
// Define Rmax
   Double_t Rmax;
   Rmax     = TMath::Sqrt(L * L + (R0 + Rm) * (R0 + Rm));
// Define R1
   Double_t R1;
   R1       = TMath::Sqrt(L * L + (R0 - Rm) * (R0 - Rm));
// Define kappa
   Double_t kappa;
   kappa    = TMath::Sqrt(1 - R1 * R1 / Rmax / Rmax);
// Define alphasq
   Double_t alphasq;
   alphasq  = 4 * R0 * Rm / (R0 + Rm) / (R0 + Rm);
// Which is my case?
   if (R0 < Rm) {
      solid_angle = 2 * PiGreco -
                    2 * L / Rmax * std::comp_ellint_1(kappa) +
                    2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(-alphasq, kappa);
   } else if (R0 == Rm) {
      solid_angle = PiGreco -
                    2 * L / Rmax * std::comp_ellint_1(kappa);
   } else if (R0 > Rm) {
      solid_angle =
         - 2 * L / Rmax * std::comp_ellint_1(kappa) +
         2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(-alphasq, kappa);
   } else {
      return 0;
   }
   return solid_angle;
}


Double_t PMSolidAngle::PMExactSolidAngleWithSign(MEGXECPMRunHeader * which_pm, Double_t * point)
{
// Computes solid angle of a point in front of a disc according to the paper of
// F. Paxton, "Solid Angle Calculation for a Circular Disk", Rev. Sci. Instr. 30 (1959) 254
//
// This version returns MINUS solid angle if the PM normal is opposite to the required point.
//
// The computation involves complete elliptic integrals of the first and third kind. The reader is referred
// to the paper for the complete explanation of the symbols, that is just sketched here:
//
// P        = The point
// Rm       = PM radius
// R0       = Distance on the disc plane from the disk center to the projection of the point P
// L        = The distance of P from the disc plane
// Rmax     = Sqrt( L**2 + ( R0 + Rm )**2 )
// R1       = Sqrt( L**2 + ( R0 - Rm )**2 )
// kappa    = Sqrt( 1 - R1**2/Rmax**2 )
// alphasq  = 4 R0 Rm / (R0 + Rm)**2
// K        = Complete elliptic function of the first kind: comp_ellint_1(k)
// Pi       = Complete elliptic function of the third kind: comp_ellint_3(alphasquare, k)
//
// The result is different for the three cases R0 < Rm, R0 = Rm, R0 > Rm
//
// Initializations
   Double_t solid_angle = 0;
   Double_t PiGreco = TMath::Pi();
//
// First of all handle special cases:
   if (false) {
      return solid_angle;
   }
// Fetch PM information:
//   -Position:
   Double_t xpm = which_pm->GetXYZAt(0);
   Double_t ypm = which_pm->GetXYZAt(1);
   Double_t zpm = which_pm->GetXYZAt(2);
//   -Normal:
   Double_t xn   = which_pm->GetDirectionAt(0);
   Double_t yn   = which_pm->GetDirectionAt(1);
   Double_t zn   = which_pm->GetDirectionAt(2);
// Fetch POINT information:
   Double_t xP   = point[0];
   Double_t yP   = point[1];
   Double_t zP   = point[2];
// Radius of photo-kathode
   Double_t Rm      = 23 * millimeter;
// Point-to-PM center distance vector (points FROM the PM TO the point)
   Double_t xd   = xP - xpm;
   Double_t yd   = yP - ypm;
   Double_t zd   = zP - zpm;
   Double_t dist = TMath::Sqrt(xd * xd + yd * yd + zd * zd);
// Point-to-disc-plane distance
   Double_t L;
   L = xn * xd + yn * yd + zn * zd;
// If the point is on the plane => the solid angle is zero.
   if (L == 0) {
      return 0;
   }
// Is this PM in shadow? If L>0 no, if L<0 yes. (remember the sign of d)
// It is defined as an integer to bear the sign of the solid angle.
   Int_t IsInShadow;
   IsInShadow = L > 0 ? +1 : -1;
// Point-projection-to-disc-center distance
   Double_t R0;
   R0       = TMath::Sqrt(dist * dist - L * L);
// Define Rmax
   Double_t Rmax;
   Rmax     = TMath::Sqrt(L * L + (R0 + Rm) * (R0 + Rm));
// Define R1
   Double_t R1;
   R1       = TMath::Sqrt(L * L + (R0 - Rm) * (R0 - Rm));
// Define kappa
   Double_t kappa;
   kappa    = TMath::Sqrt(1 - R1 * R1 / Rmax / Rmax);
// Define alphasq
   Double_t alphasq;
   alphasq  = 4 * R0 * Rm / (R0 + Rm) / (R0 + Rm);
// Which is my case?
   if (R0 < Rm) {
      solid_angle = 2 * PiGreco -
                    2 * L / Rmax * std::comp_ellint_1(kappa) +
                    2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(-alphasq, kappa);
   } else if (R0 == Rm) {
      solid_angle = PiGreco -
                    2 * L / Rmax * std::comp_ellint_1(kappa);
   } else if (R0 > Rm) {
      solid_angle =
         - 2 * L / Rmax * std::comp_ellint_1(kappa) +
         2 * L / Rmax * (R0 - Rm) / (R0 + Rm) * std::comp_ellint_3(-alphasq, kappa);
   } else {
      return 0;
   }
   return IsInShadow * solid_angle;
}

//-----
Double_t PMSolidAngle::GetCosine(MEGXECPMRunHeader* which_pm, Double_t* point)
{
   // Compute cosine of PM with respect to point
   // Make the PM to point vector
   Double_t distance[3];
   for (Int_t i = 0; i < 3; i++) {
      distance[i] = which_pm->GetXYZAt(i) - point[i];
   }
   // Makes the scalar product
   Double_t cosine;
   // computes scalar product
   cosine = which_pm->GetDirectionAt(0) * distance[0] +
            which_pm->GetDirectionAt(1) * distance[1] +
            which_pm->GetDirectionAt(2) * distance[2] ;
   // normalize scalar product
   cosine /= TMath::Sqrt(distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2]);
   // Output
   return cosine;
}

//-----
Double_t PMSolidAngle::GetDistance(MEGXECPMRunHeader* which_pm, Double_t* point)
{
   // Compute distance of PM with respect to point
   // Make the PM to point vector
   Double_t distance[3];
   for (Int_t i = 0; i < 3; i++) {
      distance[i] = which_pm->GetXYZAt(i) - point[i];
   }
   // Return distance
   return TMath::Sqrt(distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2]);
}

//-----
Double_t PMSolidAngle::PMSolidAngleApprox(MEGXECPMRunHeader* which_pm, Double_t* point)
{
   if (which_pm == NULL || point == NULL) {
      return 0;
   }
   // Get cosine
   Double_t cosine = GetCosine(which_pm, point);
   // Exit if out of sight
   if (cosine > 0) {
      return 0;
   }
   // It should see something, let's proceed
   Double_t xpm = which_pm->GetXYZAt(0);
   Double_t ypm = which_pm->GetXYZAt(1);
   Double_t zpm = which_pm->GetXYZAt(2);
   Double_t distance;
   distance = TMath::Sqrt((point[0] - xpm) * (point[0] - xpm) +
                          (point[1] - ypm) * (point[1] - ypm) +
                          (point[2] - zpm) * (point[2] - zpm));
   // At first order return Omega = cosine/dist**2
   return -cosine / distance / distance;
}
