//#include <stdio.h>      /* printf, fgets */
//#include <stdlib.h>
//include <string.h>
#include <vector>
#include <new>
#include <strstream>
#include <fstream>
//#include "clases.hpp"
//#include "clases.hpp"
using namespace std;

void Proteina::leer(char *pdbFile){

try{
  char str[5];
  str[0]='p';
  str[1]='d';
  str[2]='b';
  str[3]='s';
  str[4]='/';
  char fail[5];
  fail[0]='f';
  fail[1]='a';
  fail[2]='i';
  fail[3]='l';
  fail[4]='_';
//  x="pdbs/";
//pdbs/
//
//ifstream.open(pdbFile);
puts(pdbFile);
ifstream Fin;
//strcpy(str,pdbFile);
//strcat(str,pdbFile);
//puts(str);
Fin.open(pdbFile);
//ifstream Fin(str);
  //if(!Fin)throw printf("Cannot open PDB file \n");
if(!Fin)throw strcat(fail,pdbFile);
string buffer;
//  strcat(x,str);
  // strcat(pdbFile, "Cannot open PDB file ");
  //strcpy (str,pdbFile);
//  strcpy(Nombre_proteina, pdbFile); // give name to this protein object
int ress=0;
//char Buffer_res[5];
char Buffer_res[] = "apple";
char Buffer_cadena[]= "1";
//char Buffer_cadena;
//int poschain;
maxi_radius=0;
No_residuos=0;
char E;
char F;
No_atomos_en_prot=0;
Residuo residuobuffer;
//int Numbcad=0;
residuobuffer.No_atomos=0;
string carb("CA");
while (getline(Fin, buffer)){
if(buffer.substr(0,4)=="ATOM"){
  Atomo atomo;
  No_atomos_en_prot++;
  atomo.No_Atomo=No_atomos_en_prot;
  atomo.P_Id(buffer.substr(4,7).c_str());
  atomo.P_Tipo(buffer.substr(11,5).c_str());
  atomo.P_nombre_residuo(buffer.substr(17,3).c_str());
  atomo.P_nombre_cadena(buffer.substr(21,1).c_str());
  atomo.P_Id_residuo(buffer.substr(22,6).c_str());
  atomo.xyz(atof(buffer.substr(30,8).c_str()),atof(buffer.substr(38,8).c_str()),atof(buffer.substr(46,8).c_str()));
  atomo.Bfacfor(atof(buffer.substr(60,6).c_str()));
  atomo.P_Clase(buffer.substr(76,3).c_str());
  factores.push_back(atomo.b);
  E=atomo.Clase_atomo[1];
  F=atomo.Clase_atomo[2];
//printf("%f b factor atomo  %d \n ",atomo.b,No_atomos_en_prot);
atomo.P_radii(E,F);
if(atomo.radii>maxi_radius){
maxi_radius=atomo.radii;
}

if(atomo.Id_residuo!=ress){
if(ress==0){
cadena=1;
chain.push_back(atomo.Nombre_cadena[0]);
strcpy(Buffer_res, atomo.Nombre_residuo);
residuobuffer.Nombre_residuo(Buffer_res);
residuobuffer.P_Nombre_cadena( atomo.Nombre_cadena);
residuobuffer.P_hidrofobicidad_prev();
residuobuffer.Agregar_atomo(atomo);
residuobuffer.No_atomos++;
residuobuffer.id_cadena=cadena;

size_t found = buffer.find(carb);
if (found != string::npos) {
residuobuffer.b_carbono_central=atomo.b;
}
}else{

  if(strcmp(Buffer_cadena,atomo.Nombre_cadena )!=0){
  cadena=cadena+1;
  chain.push_back(atomo.Nombre_cadena[0]);
  Agregar_atomos_cadenas(No_atomos_en_prot);
  res_chain.push_back(No_residuos);
//  printf("%c car cadena  \n",Buffer_cadena[2] +65 );
  }
residuobuffer.apartar_memoria_res(residuobuffer.No_atomos);
//printf("cuantos atomos tiene el residuo %d \n",residuobuffer.No_atomos );
Agregar_residuo(residuobuffer);
No_residuos++;
//Residuo res;
residuobuffer.Lista_atomos.clear();
residuobuffer.No_atomos=0;
residuobuffer.Agregar_atomo(atomo);
residuobuffer.No_atomos++;
residuobuffer.Nombre_residuo(atomo.Nombre_residuo);
residuobuffer.P_Nombre_cadena( atomo.Nombre_cadena);
residuobuffer.P_hidrofobicidad_prev();
residuobuffer.P_Id_residuo(ress);
residuobuffer.id_cadena=cadena;
//printf("%s cadena buufer \n",buffer.substr(11,5).c_str() );
//printf("atomo tipo %s \n",atomo.Tipo );
//if(atomo.Tipo=="  CA "){
size_t found = buffer.find(carb);
if (found != string::npos) {
residuobuffer.b_carbono_central=atomo.b;
}
}
}else{
  residuobuffer.Agregar_atomo(atomo);
  residuobuffer.No_atomos++;
  residuobuffer.Nombre_residuo(atomo.Nombre_residuo);
residuobuffer.P_Nombre_cadena( atomo.Nombre_cadena);
  residuobuffer.P_hidrofobicidad_prev();
  residuobuffer.P_Id_residuo(ress);
residuobuffer.id_cadena=cadena;
  size_t found = buffer.find(carb);
  if (found != string::npos) {
  //if(atomo.Tipo=="  CA "){
//    printf("entro aquiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n" );
  residuobuffer.b_carbono_central=atomo.b;
  }
  if(strcmp(Buffer_cadena,atomo.Nombre_cadena )!=0){
  cadena=cadena+1;
  Agregar_atomos_cadenas(No_atomos_en_prot);
  chain.push_back(atomo.Nombre_cadena[0]);
  res_chain.push_back(No_residuos);
  }
}
//printf("Id atomo tuerno fuera %d \n", atomo.Id_residuo);
strcpy(Buffer_res, atomo.Nombre_residuo);
strcpy(Buffer_cadena, atomo.Nombre_cadena);
ress=atomo.Id_residuo ;
//  printf("Id ress fuera %d ",ress);
//printf("%s \n", buffer.c_str());
//  if(buffer.substr(0,6)=="ATOM  "){
//printf("Id ress %d ",atomo.Id_residuo);

  }
 }
 residuobuffer.P_Id_residuo(ress);
 residuobuffer.apartar_memoria_res(residuobuffer.No_atomos);
residuobuffer.P_Nombre_cadena( Buffer_cadena);
 residuobuffer.Nombre_residuo(Buffer_res);
 residuobuffer.P_hidrofobicidad_prev();
 residuobuffer.id_cadena=cadena;
  Agregar_atomos_cadenas(No_atomos_en_prot);
//     chain.push_back(Buffer_cadena[0]);
 //if(strcmp(Buffer_cadena,atomo.Nombre_cadena )!=0){
 //cadena=cadena+1;
 //}
 Agregar_residuo(residuobuffer);
 No_residuos++;
  res_chain.push_back(No_residuos);
 Fin.close();
printf("No residuos %d \n", No_residuos);

//homologar_memory
int prue=0;
for (int i = 0; i < No_residuos; ++i){
  prue=Lista_residuos[i].No_atomos;
  for (int j = 0; j < prue; ++j){
    Lista_residuos[i].atomos_en_residuo[j]= Lista_residuos[i].Lista_atomos[j];
      Lista_residuos[i].atomos_en_residuo[j].apartar_memoria_3();
for (int g = 0; g <200; ++g){
    Lista_residuos[i].atomos_en_residuo[j].ids[g]=0;
}

  }
  apartar_memoria_proteina(No_residuos);
for (int k = 0; k < No_residuos; ++k){
residuos_en_proteina[k]=Lista_residuos[k];
}
}

}
 catch(char* pMsg) { cerr << endl << "Exception:" << pMsg << endl; }

}
