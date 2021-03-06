#include <iostream>
#include "clases.hpp"
#include "lectura.hpp"
#include "adft.hpp"
//#include "conservation.hpp" (if avaialbe put cons. score)
#include "bfactos.hpp"
#include <stdbool.h>
#include <time.h>

using namespace std;

int main(){
  int aidi;
  float area,a,b,c;
  float hidro;
  int contactos;
  float prev;
  float consv;
  float bfac,bf;
  float energies;
  char cc='"';
  int id;
  bool interface;
clock_t start, stop;
double totalTime;
Proteina g;
float punta=1.4;
int puntos=1000;
float suma=0;
char r[]="1es7FH.pdb";
g.leer(r);
//printf("sali lectura \n");
//g.scores(r);  (if available, we recover conservation score at each chain in separe files)
g.exp_dft(punta,r);
//auto t1 = std::chrono::high_resolution_clock::now();
start = clock();
printf("probe %f with number of points: %d \n",punta,puntos);
g.asa(punta,suma,puntos);
stop = clock();
totalTime = (stop - start) / ((double)CLOCKS_PER_SEC/1000);
//auto t2 = std::chrono::high_resolution_clock::now();
//auto duration = duration_cast<miliseconds>( t2 - t1 ).count();
printf("parallel ASA  method took (ms): %f \n",totalTime );
g.flexibility(r);
printf("with chains: %d \n",g.cadena );
g.impr(r);
g.interface();
ofstream outfile ("data_generated.txt");
outfile << "#datosproteina:"<<r<<endl;
outfile << "#Relevan: No. of chains: "<<g.cadena<<" Residue number :" <<g.No_residuos<<" No. of atoms "<<g.No_atomos_en_prot<<endl;
outfile <<"#WITHOUT HOTREGION"<<endl;
outfile <<"#WITHOUT DDG"<<endl;
//if availabe put the energies calculated by DFT
for(int w=0; w< g.No_residuos;w++){
aidi=g.residuos_en_proteina[w].Id_Res;
area=g.residuos_en_proteina[w].area_res;
hidro=g.residuos_en_proteina[w].hidrofobicidad;
contactos=g.residuos_en_proteina[w].contactos1;
prev=g.residuos_en_proteina[w].prevalencia;
consv=g.residuos_en_proteina[w].conservation_score;
bfac=g.residuos_en_proteina[w].b_factor;
interface=g.residuos_en_proteina[w].interfaz;
//energies=g.residuos_en_proteina[w].energias_dft;
//a=g.residuos_en_proteina[w].atomos_en_residuo[0].x;
//b=g.residuos_en_proteina[w].atomos_en_residuo[0].y;
//c=g.residuos_en_proteina[w].atomos_en_residuo[0].z;
//id=g.residuos_en_proteina[w].atomos_en_residuo[0].Id;
//bf=g.residuos_en_proteina[w].atomos_en_residuo[0].b;

outfile <<"["<<cc<<g.residuos_en_proteina[w].Nombre_cad<<cc<<","<<cc<<g.residuos_en_proteina[w].Nombre_res<<cc<<","<<aidi<<","<<bfac<<","<<area<<","<<hidro<<","<<prev<<","<<consv<<","<<energies<<","<<interface<<",0],"<<endl;
//printf("Chain %s Residue %s id %d  b factor %.3f area %.3f hidrof. %.2f,atomic contacts %d prevalence %.2f score conserv. %.3f ground-state energies %.3f if belong to interface %d \n",g.residuos_en_proteina[w].Nombre_cad,g.residuos_en_proteina[w].Nombre_res,aidi ,bfac,area ,hidro,contactos,prev,consv,energies,interface );
//printf("Coordenadas de su atomo 1 x %.3f y %.3f z %.3f , tipo %s,id %d,factorb %.3f, elemento %s \n",a,b,c,g.residuos_en_proteina[w].atomos_en_residuo[0].Tipo,id,bf,g.residuos_en_proteina[w].atomos_en_residuo[0].Clase_atomo);
}

printf("total area sum %f \n",suma );
g.clear();
return 0;
}
