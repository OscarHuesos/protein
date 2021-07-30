#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <new>
//#include <string>

using namespace std;
void Proteina::exp_dft(float punta,char *pdbFile){

int xx,yy,zz;
  char backslash='\\';
  char cc='"';
ofstream outfile ("datap.py");

outfile << "#my text here! " << endl;
outfile << "#my text here! " << endl;
outfile << "#"<<pdbFile<< endl;

outfile << "mol=[]" << endl;
//outfile <<cc<<cc<<cc<< endl;
//int q=residuos_en_proteina[0].Id_Res;
float dmax=2.0*(punta+maxi_radius);
for (int i=0; i < No_residuos; ++i){
//printf("Id residuo %d \n", residuos_en_proteina[i].Id_Res);
outfile <<residuos_en_proteina[i].Nombre_res<<residuos_en_proteina[i].Id_Res<<
"_dat="<<cc<<cc<<cc<<backslash<<endl;
for (int j=0; j < residuos_en_proteina[i].No_atomos; ++j){
  xx=floor(residuos_en_proteina[i].atomos_en_residuo[j].x/dmax);
  yy=floor(residuos_en_proteina[i].atomos_en_residuo[j].y/dmax);
  zz=floor(residuos_en_proteina[i].atomos_en_residuo[j].z/dmax);
  residuos_en_proteina[i].atomos_en_residuo[j].P_caja(xx,yy,zz);

  if((i==0) && (j==0) ){
  Boxx buffer(xx,yy,zz);
  Lista_Cajas.push_back(buffer);
  Lista_Cajas[0].Agregar_Atm(residuos_en_proteina[i].atomos_en_residuo[j]);
  }else{
  it = find_if(Lista_Cajas.begin(), Lista_Cajas.end(), Boxx::Finder (xx,yy,zz));
   if (it != Lista_Cajas.end()) {
   auto int idx = distance(Lista_Cajas.begin(), it);
  // int idx = distance(Lista_Cajas.begin(), it);
   Lista_Cajas[idx].Agregar_Atm(residuos_en_proteina[i].atomos_en_residuo[j]);
  }else{
    Boxx buffer(xx,yy,zz);
    Lista_Cajas.push_back(buffer);
    Lista_Cajas.back().Agregar_Atm(residuos_en_proteina[i].atomos_en_residuo[j]);
  }
  }

outfile <<residuos_en_proteina[i].atomos_en_residuo[j].Clase_atomo<<"   "<<
residuos_en_proteina[i].atomos_en_residuo[j].x<<"  "<<
residuos_en_proteina[i].atomos_en_residuo[j].y<<"  "<<
residuos_en_proteina[i].atomos_en_residuo[j].z<<endl;
}
outfile <<cc<<cc<<cc<<endl;
outfile <<residuos_en_proteina[i].Nombre_res<<residuos_en_proteina[i].Id_Res<<
" = read_xyz_lines("<<residuos_en_proteina[i].Nombre_res<<residuos_en_proteina[i].Id_Res<<
"_dat.splitlines(),name="<<cc<<residuos_en_proteina[i].Nombre_res<<
residuos_en_proteina[i].Id_Res<<cc<<")"<<endl;
outfile <<"mol.append("<<residuos_en_proteina[i].Nombre_res<<residuos_en_proteina[i].Id_Res<<
") \n"<<endl;
d_max=dmax;
}
outfile.close();
}
