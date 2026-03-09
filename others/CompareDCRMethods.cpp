//______________________________________________________________________________
#include "macros/xec/EnergyScaleCalib/common/Pi0OpeningAngleCorrection.h"
#include "ROMETreeInfo.h"
#include "xec/xectools.h"

//______________________________________________________________________________
TH1D* GetEGammaHisto(TString filename)
{
   std::cout << "Start read " << filename << std::endl;

   TH1D* h = new TH1D(Form("h_%s", filename.Data()),
                      "; #it{E}_{#gamma} (GeV); Entries / (100 keV)", 600, 0.04, 0.1);

   //auto fin = new TFile(filename);
   //auto rec = (TTree*)fin->Get("rec");

   auto rec = new TChain("rec");
   rec->Add(filename);

   TTreeReader reader(rec);
   TTreeReaderValue<MEGRecData> recoRV(reader, "reco.");
   //TTreeReaderArray<MEGBGOCEXResult> bgocexresultRA(reader, "bgocexresult");

   while (reader.Next()) {
      if (recoRV->GetEvstatGammaAt(0) >= 2) continue;
      //if (bgocexresultRA.GetSize() != 1) continue;

      //auto corr = GetOpeningAngleCorrection(bgocexresultRA.At(0).GetopeningAngle(), true);
      //h->Fill(recoRV->GetEGammaAt(0) * corr);
      h->Fill(recoRV->GetEGammaAt(0));
   }
   //fin->Close();
   //delete(fin);
   SafeDelete(rec);
   std::cout << "Finish reading " << filename << std::endl;
   return h;
}


//______________________________________________________________________________
void CompareDCRMethods(TString frec1name = "recreduced_cex2022_dcrtest.root", TString frec2name = "recreduced_cex2022_20240625.root")
{
   gStyle->SetOptStat(0);
   gStyle->SetOptFit(1111);

   TCanvas* c = new TCanvas("c", "c");
   auto h1 = GetEGammaHisto(frec1name);
   auto h2 = GetEGammaHisto(frec2name);
   h2->SetLineColor(kBlue);

   h1->Draw();
   h2->Draw("sames");

   TF1* f = new TF1("f", XECTOOLS::ExpGaus, 0.04, 0.06, 4);
   f->SetLineColor(kRed);
   //f->SetParameters(h1->GetMaximum(), 0.055, 0.02 * 0.055, -0.001 * 0.055);
   f->SetParameters(h1->GetMaximum(), 0.0528, 0.01 * 0.0528, -0.001 * 0.0528);
   auto res1 = h1->Fit(f, "SL", "", 0.052, 0.0535);
   //res1->Print();
   auto res2 = h2->Fit(f, "SL", "", 0.052, 0.0535);
   //res2->Print();
}

