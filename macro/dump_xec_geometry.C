// dump_xec_geometry.C
//
// Dump XECPMRunHeader geometry (ID, is_sipm, face, xyz, direction) from a
// rec file to a plain text file. Used to verify that our Python
// lib/sensor_directions.txt matches what meganalyzer uses at runtime.
//
// Output format (whitespace-separated, header comment line):
//   # id is_sipm face x y z dir_x dir_y dir_z
//   0 1 0 -29.8508 57.4104 -32.4601 -0.4575 -0.0753 0.1305
//   1 ...
//
// Usage:
//   cd $MEG2SYS/analyzer
//   ./meganalyzer -b -q \
//       'path/to/macro/dump_xec_geometry.C("rec559261.root", "out.txt")'

#include <fstream>
#include <iostream>
#include <TFile.h>
#include <TClonesArray.h>

#if !defined(__CLING__) || defined(__ROOTCLING__)
#  include "include/generated/MEGXECPMRunHeader.h"
#endif

void dump_xec_geometry(const char* recFile, const char* outFile) {
    TFile *f = TFile::Open(recFile);
    if (!f || f->IsZombie()) {
        std::cerr << "[dump_xec_geometry] Cannot open " << recFile << std::endl;
        return;
    }

    TClonesArray *hdrs = (TClonesArray*)f->Get("XECPMRunHeader");
    if (!hdrs) {
        std::cerr << "[dump_xec_geometry] XECPMRunHeader not found in "
                  << recFile << std::endl;
        f->Close();
        return;
    }

    std::ofstream ofs(outFile);
    if (!ofs.is_open()) {
        std::cerr << "[dump_xec_geometry] Cannot write " << outFile << std::endl;
        f->Close();
        return;
    }

    ofs << "# id is_sipm face x y z dir_x dir_y dir_z\n";

    const Int_t n = hdrs->GetEntriesFast();
    Int_t nWritten = 0;
    for (Int_t i = 0; i < n; i++) {
        MEGXECPMRunHeader *h = (MEGXECPMRunHeader*)hdrs->At(i);
        if (!h) continue;
        ofs << i << " "
            << h->GetIsSiPM() << " "
            << h->GetFace() << " "
            << h->GetXYZAt(0) << " "
            << h->GetXYZAt(1) << " "
            << h->GetXYZAt(2) << " "
            << h->GetDirectionAt(0) << " "
            << h->GetDirectionAt(1) << " "
            << h->GetDirectionAt(2) << "\n";
        nWritten++;
    }

    ofs.close();
    f->Close();
    std::cout << "[dump_xec_geometry] Wrote " << nWritten << " sensors to "
              << outFile << std::endl;
}
