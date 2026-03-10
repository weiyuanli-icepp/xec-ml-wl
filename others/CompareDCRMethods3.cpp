//______________________________________________________________________________
#if !defined(__CLING__) || defined(__ROOTCLING__)
#   include "ROMETreeInfo.h"
#   include "include/generated/MEGRecData.h"
#   include "xec/xectools.h"
#else
    class ROMETreeInfo;
    class MEGRecData;
#endif

// ExpGaus: Gaussian with exponential low-energy tail
// par[0]: height, par[1]: peak, par[2]: sigma, par[3]: transition point
Double_t ExpGausLocal(Double_t *x, Double_t *par)
{
   Double_t fitval;
   if (x[0] > par[1] + par[3]) {
      if (par[2] != 0)
         fitval = par[0] * TMath::Exp(-1 * (x[0] - par[1]) * (x[0] - par[1]) / 2 / par[2] / par[2]);
      else
         fitval = 0;
   } else {
      if (par[2] != 0)
         fitval = par[0] * TMath::Exp(par[3] / par[2] / par[2] * (par[3] / 2 - (x[0] - par[1])));
      else
         fitval = 0;
   }
   return fitval;
}

//______________________________________________________________________________
TH1D* GetEGammaHisto(TString name, TString filepattern)
{
   std::cout << "Start reading " << name << ": " << filepattern << std::endl;

   TH1D* h = new TH1D(Form("h_%s", name.Data()),
                       "; #it{E}_{#gamma} (GeV); Entries / (100 keV)", 600, 0.04, 0.1);

   auto rec = new TChain("rec");
   rec->Add(filepattern);
   Long64_t nFiles = rec->GetNtrees();
   Long64_t nEntries = rec->GetEntries();
   std::cout << "  " << nFiles << " files, " << nEntries << " entries" << std::endl;

   TTreeReader reader(rec);
   TTreeReaderValue<MEGRecData> recoRV(reader, "reco.");

   while (reader.Next()) {
      if (recoRV->GetEvstatGammaAt(0) >= 2) continue;
      h->Fill(recoRV->GetEGammaAt(0));
   }
   SafeDelete(rec);
   std::cout << "  " << h->GetEntries() << " entries filled" << std::endl;
   return h;
}

//______________________________________________________________________________
void CompareDCRMethods3(
   TString baseDir = "/data/project/meg/shared/subprojects/xec/li_w/cex/2023")
{
   gStyle->SetOptStat(0);

   TString saPattern  = Form("%s/sa/rec*.root",  baseDir.Data());
   TString avgPattern = Form("%s/avg/rec*.root", baseDir.Data());
   TString inpPattern = Form("%s/inp/rec*.root", baseDir.Data());

   auto hSa  = GetEGammaHisto("sa",  saPattern);
   auto hAvg = GetEGammaHisto("avg", avgPattern);
   auto hInp = GetEGammaHisto("inp", inpPattern);

   hSa->SetLineColor(kBlack);
   hAvg->SetLineColor(kBlue);
   hInp->SetLineColor(kRed);

   hSa->SetLineWidth(2);
   hAvg->SetLineWidth(2);
   hInp->SetLineWidth(2);

   // Fit each with ExpGaus
   TF1* fSa  = new TF1("fSa",  ExpGausLocal, 0.04, 0.06, 4);
   TF1* fAvg = new TF1("fAvg", ExpGausLocal, 0.04, 0.06, 4);
   TF1* fInp = new TF1("fInp", ExpGausLocal, 0.04, 0.06, 4);

   fSa->SetLineColor(kBlack);   fSa->SetLineStyle(2);
   fAvg->SetLineColor(kBlue);   fAvg->SetLineStyle(2);
   fInp->SetLineColor(kRed);    fInp->SetLineStyle(2);

   Double_t fitLo = 0.052, fitHi = 0.0535;

   fSa->SetParameters(hSa->GetMaximum(), 0.0528, 0.01 * 0.0528, -0.001 * 0.0528);
   auto resSa = hSa->Fit(fSa, "SL", "", fitLo, fitHi);

   fAvg->SetParameters(hAvg->GetMaximum(), 0.0528, 0.01 * 0.0528, -0.001 * 0.0528);
   auto resAvg = hAvg->Fit(fAvg, "SL", "", fitLo, fitHi);

   fInp->SetParameters(hInp->GetMaximum(), 0.0528, 0.01 * 0.0528, -0.001 * 0.0528);
   auto resInp = hInp->Fit(fInp, "SL", "", fitLo, fitHi);

   // Draw
   TCanvas* c = new TCanvas("c", "DCR Method Comparison", 800, 600);

   Double_t ymax = 1.15 * TMath::Max(hSa->GetMaximum(),
                          TMath::Max(hAvg->GetMaximum(), hInp->GetMaximum()));
   hSa->SetMaximum(ymax);

   hSa->Draw("hist");
   hAvg->Draw("hist same");
   hInp->Draw("hist same");

   fSa->Draw("same");
   fAvg->Draw("same");
   fInp->Draw("same");

   // Legend with fit results
   auto leg = new TLegend(0.50, 0.55, 0.88, 0.88);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetTextSize(0.035);

   TString names[3]        = {"Solid angle", "Average", "ML inpaint"};
   TFitResultPtr results[3] = {resSa, resAvg, resInp};
   TH1D* histos[3]         = {hSa, hAvg, hInp};

   for (int i = 0; i < 3; i++) {
      Double_t mean = results[i]->Parameter(1) * 1e3;
      Double_t reso = (results[i]->Parameter(1) != 0)
                      ? results[i]->Parameter(2) / results[i]->Parameter(1) * 100 : 0;
      leg->AddEntry(histos[i],
                    Form("%s: #mu=%.2f MeV, #sigma/E=%.2f%%", names[i].Data(), mean, reso), "l");
   }
   leg->Draw();

   // Print summary
   std::cout << "\n=== DCR Method Comparison ===" << std::endl;
   for (int i = 0; i < 3; i++) {
      Double_t mean  = results[i]->Parameter(1) * 1e3;
      Double_t sigma = results[i]->Parameter(2) * 1e3;
      Double_t reso  = results[i]->Parameter(2) / results[i]->Parameter(1) * 100;
      std::cout << Form("  %-12s: mean = %.3f MeV, sigma = %.3f MeV, sigma/E = %.2f%%",
                         names[i].Data(), mean, sigma, reso) << std::endl;
   }

   c->SaveAs("CompareDCRMethods3.pdf");
}
