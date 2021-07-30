#include <TROOT.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TLatex.h>
#include "TString.h"
#include "TSQLServer.h"
#include "TSQLRow.h"
#include "TSQLResult.h"
#include <TPaveText.h>
#include <THStack.h>
#include "TH1.h"
#include "TH2.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TText.h"
#include "TF1.h"
#include "TMath.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TApplication.h"



//root method

void Rootgraph(int nsample,double *v, char *save,int maxx,char *title){
int i;
int minx=0;
//int maxx=45;
TH1* h1 = new TH2D("H1", "M", 60, minx,maxx ,5, 0, 2);
//--histogram and box
TH1* h2 = new TH1D("H2", "Medics", 60, minx, maxx);

for (i = 0; i < nsample; i++) {
       h1->Fill(v[i], 1);
         h2->Fill(v[i]);
}

  TCanvas *c1 = new TCanvas("c", "multipads", 1000, 1000);
  TPad *p1 = new TPad("p1", "p1", 0, 0.7, 1, 1, 0, 0, 0);
  p1->SetBottomMargin(0);
  p1->Draw();
  TPad *p2 = new TPad("p2", "p2", 0, 0, 1, 0.7, 0, 0, 0);
  p2->SetTopMargin(0);
  p2->Draw();
//tedence variables
  Double_t Std = h2->GetStdDev();
  Double_t E = h2->GetEntries();
  Double_t Mean = h2->GetMean();
  Double_t K = h2->GetKurtosis();
  Double_t G2=h2->GetMeanError();
  Double_t Sk=h2->GetSkewness();
  Double_t down = h2->GetBinContent(0);
  Double_t up = h2->GetBinContent(h2->GetNbinsX() + 1);

  h2->Scale(1/E);
  h2->SetLineColor(kRed);
  h2->SetBarWidth(0.8);
  h2->SetBit( TH1::kNoTitle, true );
  h2->SetFillColor(4);

  char array[100];
  TPaveStats *ptstats = new TPaveStats(0.7,0.6,0.9,0.9,"brNDC");
  ptstats->SetName("stats");
  ptstats->SetBorderSize(1);
  ptstats->SetFillColor(0);
  ptstats->SetTextAlign(12);
  ptstats->SetTextFont(42);

  TText *ptstats_LaTex = ptstats->AddText("H");
  ptstats_LaTex->SetTextSize(0.0368);
  sprintf(array, "Entries = %f", E);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Mean= %f", Mean);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Standard Dev. = %f", Std);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Skewness = %f", Sk);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Kurtosis=  %f", K);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Mean Error = %f", G2);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Overflow = %f", up);
  ptstats_LaTex = ptstats->AddText(array);
  sprintf(array, "Underflow = %f", down);
  ptstats_LaTex = ptstats->AddText(array);
  gStyle->SetOptStat(0);

  TPaveText *pt = new TPaveText(0.2,0.7,0.8,0.93,"blNDC");
  pt->SetName("title");
  pt->SetBorderSize(0);
  pt->SetFillColor(0);
  pt->SetFillStyle(0);
  pt->SetTextFont(42);
  TText *pt_LaTex = pt->AddText(title);

  //----boxplot--------------
  TAxis* a = h1->GetYaxis();
  a->SetNdivisions(-2);
  p1->cd();
  gPad->SetTicky(0);
  h1->SetLineColor(4);
  h1->SetBit( TH1::kNoTitle, true );
  h1->Draw("candley3");
  pt->Draw();
  p2->cd();
  h2->Draw("histbar");
  ptstats->Draw();

   c1->Update();
   c1->SetGrid();
   c1->SaveAs(save);

  }
