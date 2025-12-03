#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TLine.h"
#include "TF1.h"

void plot_residuals(const char* csv_file = "artifacts/run_cv2_test_01/predictions_run_cv2_test_01.csv", bool SaveFlag = true) {

    // 1. Setup Histograms
    // 1D Residuals
    TH1F* h_theta = new TH1F("h_theta", "#theta Residuals; #theta_{pred} - #theta_{true} [deg]; Events", 300, -60, 60);
    TH1F* h_phi   = new TH1F("h_phi",   "#phi Residuals; #phi_{pred} - #phi_{true} [deg]; Events",     300, -60, 60);

    // 2D Heatmaps (Truth vs Pred)
    // Adjust ranges based on expected physics range (e.g. 0-180 for theta, -180-180 for phi)
    // Here using auto-range or reasonable defaults.
    TH2F* h2_theta = new TH2F("h2_theta", "#theta Regression; #theta_{true} [deg]; #theta_{pred} [deg]", 100, 0, 180, 100, 0, 180);
    TH2F* h2_phi   = new TH2F("h2_phi",   "#phi Regression; #phi_{true} [deg]; #phi_{pred} [deg]",     100, -95, 95, 100, -95, 95);

    // 2. Read CSV
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << csv_file << std::endl;
        return;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    // Columns: true_theta, true_phi, pred_theta, pred_phi
    double t_th, t_ph, p_th, p_ph;
    
    // Track min/max for auto-ranging lines if needed, but fixed range is usually safer for angles
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val_str;
        std::vector<double> row;

        while (std::getline(ss, val_str, ',')) {
            row.push_back(std::stod(val_str));
        }

        if (row.size() < 4) continue;

        t_th = row[0];
        t_ph = row[1];
        p_th = row[2];
        p_ph = row[3];

        h_theta->Fill(p_th - t_th);
        h_phi->Fill(p_ph - t_ph);
        
        h2_theta->Fill(t_th, p_th);
        h2_phi->Fill(t_ph, p_ph);
    }
    file.close();

    // 3. Draw
    // Disable general stats, Enable Fit stats
    gStyle->SetOptStat(0);     // Hide standard stats box (Entries, Mean, RMS)
    gStyle->SetOptFit(1111);   // Show Fit parameters (Chi2, Prob, p0, p1, p2...)
    gStyle->SetPalette(kBird); // Good color palette
    
    // --- DISABLE GRIDS EXPLICITLY ---
    gStyle->SetPadGridX(0);
    gStyle->SetPadGridY(0);
    
    TCanvas* c1 = new TCanvas("c1", "Analysis", 1200, 1000);
    c1->Divide(2, 2);

    // Top Left: Theta Residuals
    c1->cd(1);
    h_theta->SetLineColor(kBlue+1);
    h_theta->Fit("gaus"); // Fit with Gaussian
    TF1 *f_theta = h_theta->GetFunction("gaus");
    if (f_theta) {
        f_theta->SetLineColor(kRed);
        f_theta->SetLineWidth(2);
    }
    h_theta->Draw();

    // Top Right: Phi Residuals
    c1->cd(2);
    h_phi->SetLineColor(kBlue+1);
    h_phi->Fit("gaus");   // Fit with Gaussian
    TF1 *f_phi = h_phi->GetFunction("gaus");
    if (f_phi) {
        f_phi->SetLineColor(kRed);
        f_phi->SetLineWidth(2);
    }
    h_phi->Draw();

    // Bottom Left: Theta Heatmap
    c1->cd(3);
    gPad->SetLogz(); // Log scale for Z (density)
    h2_theta->Draw("COLZ");
    TLine* l_th = new TLine(0, 0, 180, 180);
    l_th->SetLineStyle(2); l_th->SetLineColor(kRed);
    l_th->Draw();

    // Bottom Right: Phi Heatmap
    c1->cd(4);
    gPad->SetLogz();
    h2_phi->Draw("COLZ");
    TLine* l_ph = new TLine(-95, -95, 95, 95);
    l_ph->SetLineStyle(2); l_ph->SetLineColor(kRed);
    l_ph->Draw();

    // 4. Save
    if (SaveFlag) {
      TString outName = TString(csv_file).ReplaceAll(".csv", "_analysis.pdf");
      c1->SaveAs(outName);
      std::cout << "Saved analysis plot to: " << outName << std::endl;
    }
}