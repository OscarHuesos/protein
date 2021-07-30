//#include "gpu.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <new>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <math.h>

//Definir las clases
using namespace std;

class Atomo{
public:
char  Tipo[6], isomero[2], Nombre_residuo[5], Nombre_cadena[3],Clase_atomo[3];
  float x;
  float y;
  float z;
  float b;
  float conservation;
  int caja[3];
  float radii;
  float area_atomo;
  int No_Atomo;
  int No_Atomos2;
  int  chain_id , No_enlaces ,Id_residuo , Id; // sequential numbers
  float *xs;
  float *ys;
  float *zs;
  float *radd;
  int *ids;
  int *ids2;
  vector<Atomo> Lista_Vecinos;
  bool interfaz=false;

  Atomo() {};
 ~Atomo() {}
 void P_Id(const char *str) {
   Id=atoi(str);
   }
  void cadena_identificador(int I) {
    chain_id=I;
    }
    void P_Id_residuo(const char *str) {
      Id_residuo=atoi(str);
      }
  void P_Tipo(const char *str) {
     strcpy(Tipo, str);
   }
  void P_nombre_residuo(const char *str) {
     strcpy( Nombre_residuo, str);
   }
   void P_nombre_cadena(const char *str) {
        strcpy( Nombre_cadena, str);
      }
      void P_Clase(const char *str){
        strcpy(Clase_atomo, str);
      }

  void P_caja(int a, int b, int c){
     caja[0]=a;
     caja[1]=b;
     caja[2]=c;
    }

    void xyz(const float a, const float b, const float c){
       x=a;
       y=b;
       z=c;
      }

void P_radii(char e1, char e2) {
radii=0;

if ( (e2!='l') ||  (e2!='u') ||  (e2!='n')  ||  (e2!='g')  ||   (e2!='a')){

switch(e1){
case 'H': radii=1.20; break;
case 'N': radii=1.55; break;
case 'O': radii=1.52; break;
case 'C': radii=1.7;  break;
case 'I': radii=1.98; break;
case 'P': radii=1.8;  break;
case 'B': radii=1.85; break;
case 'S': radii=1.8;  break;
case 'F': radii=1.47; break;
case 'K': radii=2.75; break;
}
}else{

switch(e1){
   case 'C':
switch(e2){
   case 'l': radii=1.75; break;
   case 'u': radii=1.4;  break;
}
break;
case 'M':
switch(e2){
   case 'n': radii=1.73;  break;
   case 'g': radii=1.73;  break;
}
break;
case 'N':
switch(e2){
   case 'a': radii=2.27; break;
}
}
}
}

void Bfacfor(float t) { b=t; }
void P_conservation(float t) { conservation=t; }
void vecinos();
void apartar_memoria_3();
};


class Residuo{
//porqu valida cuantico a area
public:
  char Nombre_res[4], Nombre_cad[3];
  float hidrofobicidad;
  float area_res;
  float prevalencia;
  float conservation_score;
  float b_carbono_central;
  float b_factor;
  int id_cadena;
  float energias_dft;
  char	Cadena_del_res[2];
  int Id_Res;
  int No_atomos;
  int contactos1=0;
  Atomo *atomos_en_residuo;
  vector<Atomo> Lista_atomos;
  int  total_de_atomos;
  bool interfaz=false;
  bool hot_spot;
  Residuo() {};
  ~Residuo() {};
  void Nombre_residuo(const char *str) {
  strcpy( Nombre_res, str); }
  void P_Nombre_cadena(const char *str) {
  strcpy( Nombre_cad, str); }
  void P_Id_residuo(int i) { Id_Res=i; }
  void Agregar_atomo(Atomo & newatm){
  Lista_atomos.push_back(newatm);
  }

  void P_hidrofobicidad_prev() {
  if(strcmp(Nombre_res, "ALA")==0){  hidrofobicidad=1.8; prevalencia=10 ; }
  if(strcmp(Nombre_res, "LYS")==0){  hidrofobicidad=-3.9; prevalencia= 6.29; }
  if(strcmp(Nombre_res, "LEU")==0){ hidrofobicidad=3.8;prevalencia= 0.83 ; }
  if(strcmp(Nombre_res, "TYR")==0){ hidrofobicidad=-1.3;prevalencia=12.3;}
  if(strcmp(Nombre_res, "ARG")==0){ hidrofobicidad=-4.5;prevalencia=13.3;}
  if(strcmp(Nombre_res, "ASN")==0){ hidrofobicidad=-3.5;prevalencia=5.05;}
  if(strcmp(Nombre_res, "ASP")==0){ hidrofobicidad=-3.5;prevalencia=9.04;}
  if(strcmp(Nombre_res, "CYS")==0){ hidrofobicidad=2.5;prevalencia=0;}
  if(strcmp(Nombre_res, "GLN")==0){ hidrofobicidad=-3.5;prevalencia=3.13;}
  if(strcmp(Nombre_res, "GLU")==0){ hidrofobicidad=-3.5;prevalencia=3.64;}
  if(strcmp(Nombre_res, "HIS")==0){ hidrofobicidad=-3.2;prevalencia=8;}
  if(strcmp(Nombre_res, "GLY")==0){ hidrofobicidad=-0.4;prevalencia=3.57;}
  if(strcmp(Nombre_res, "ILE")==0){ hidrofobicidad=4.5;prevalencia=9.62;}
  if(strcmp(Nombre_res, "MET")==0){ hidrofobicidad=1.9;prevalencia=2.9;}
  if(strcmp(Nombre_res, "PHE")==0){ hidrofobicidad=2.8;prevalencia=3.01;}
  if(strcmp(Nombre_res, "PRO")==0){ hidrofobicidad=-1.6;prevalencia=6.74;}
  if(strcmp(Nombre_res, "SER")==0){ hidrofobicidad=-0.8;prevalencia=1.12;}
  if(strcmp(Nombre_res, "THR")==0){ hidrofobicidad=-0.7;prevalencia=1.53;}
  if(strcmp(Nombre_res, "TRP")==0){ hidrofobicidad=-0.9;prevalencia=21.05;}
  if(strcmp(Nombre_res, "VAL")==0){ hidrofobicidad=4.2;prevalencia=0;}
  //  case 'LYS': hidrofobicidad=-3.9;
  //  case 'LEU': hidrofobicidad=3.8;
  //  case 'TYR': hidrofobicidad=-1.3;
  //  case 'ARG': hidrofobicidad=-4.5;
  //  case 'ASN': hidrofobicidad=-3.5;
  //  case 'ASP': hidrofobicidad=-3.5;
  //  case 'CYS': hidrofobicidad=2.5;
  //  case 'GLN': hidrofobicidad=-3.5;
//    case 'GLU': hidrofobicidad=-3.5;
//    case 'HIS': hidrofobicidad=-3.2;
//    case 'GLY': hidrofobicidad=-0.4;
//    case 'ILE': hidrofobicidad=4.5;
//    case 'MET': hidrofobicidad=1.9;
//    case 'PHE': hidrofobicidad=2.8;
//    case 'PRO': hidrofobicidad=-1.6;
  //  case 'SER': hidrofobicidad=-0.8;
  //  case 'THR': hidrofobicidad=-0.7;
  //  case 'TRP': hidrofobicidad=-0.9;
  //  case 'VAL': hidrofobicidad=4.2;
  }
  void P_No_atomos(int i) {
     total_de_atomos=i;
   }
  void P_Id_res(int i) { Id_Res=i; }
  void setParm();
  void apartar_memoria_res(int n) ;
};

class Boxx{

public:
int x,y,z;
int id;
int No_atoms=0;
Atomo *atomos_caja;
vector<Atomo> Lista_atomos_caja;
Boxx(int t,int tt, int ttt) {
  x=t;
  y=tt;
  z=ttt;
};
~Boxx() {};
void Agregar_Atm( Atomo & atm){
Lista_atomos_caja.push_back(atm);
}
void Add_No(int a){
 No_atoms=a;
}
void apartar_memoria_at(int n) ;
struct Finder {
    Finder(int const & a,int const &b,int const &c) : xx(a),yy(b),zz(c) {
    }
    bool operator () (const Boxx & el) const {
         return ( (el.x == xx) && (el.y == yy) &&  (el.z == zz) ) ;
     }
    int xx;
    int yy;
     int zz;
};
};


class Proteina{

public:
float d_max;
float maxi_radius;
  char Nombre_proteina[80];
  int No_residuos;
  int No_atomos_en_prot;
  float area_total;
  int cadena;
///vectors////////////////////////////////
vector<char>  chain;
vector<int>  res_chain;
vector<int>  cadens;
 vector<int>  No_atomos_por_cadena;
  vector<float> bmean;
  vector<Atomo> Lista_atomos_proteina;
  vector<Residuo> Lista_residuos;
  vector<Boxx> Lista_Cajas;
  vector<Boxx>::iterator it;
  vector<float> factores;
//apuntadores/////////////////////////////////////////////
Atomo *atomos_en_proteina;
Residuo *residuos_en_proteina;
Boxx *Cajas;
  Proteina() {};
  ~Proteina() {};

void Agregar_residuo( Residuo & res){
//  No_residuos++;
 Lista_residuos.push_back(res);
}
void    Agregar_atomos_cadenas( int a){
cadens.push_back(a);
}
void scores(char *file);
void exp_dft(float punta,char *pdbFile);
void leer(char *file);
void apartar_memoria_proteina(int n);
void apartar_memoria_cajas(int n);
void destroy_list_obj();
void clear();
void asa(float punta,float& suma,int puntos);
void interface();
void apartar_lista_atomos(int n);
void clear_interface_list();
void destroy_list_obj_2();
void impr(char *pdbFile){
  ofstream outfile ("datasa.txt");
  float res=0;
  float hidro=0;
  outfile <<pdbFile<< endl;
  for(int w=0; w< No_residuos;w++){
  res=residuos_en_proteina[w].area_res;
  hidro=residuos_en_proteina[w].hidrofobicidad;
  outfile <<res<<"        "<<hidro<< endl;
  }
  outfile.close();
}


void energias(char *dfts){

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

puts(dfts);
ifstream Fin;
int re=0;
float energy=0;
char Buffer_res[3];
char id_res[3];
char ener[10];
Fin.open(dfts);
int con=0;
if(!Fin)throw strcat(fail,dfts);
string buffer;
while (getline(Fin, buffer)){
if(buffer.substr(0,1)=="["){

strcpy(id_res, buffer.substr(4,3).c_str());
strcpy(Buffer_res, buffer.substr(1,3).c_str());
re= atoi(id_res);

if(re>99){
  strcpy(ener, buffer.substr(8,10).c_str());
}else{
strcpy(ener, buffer.substr(7,10).c_str());
}
energy=atof(ener);

for(int w=0; w< No_residuos;w++){
if((strcmp(residuos_en_proteina[w].Nombre_res,Buffer_res)==0)&&(residuos_en_proteina[w].Id_Res==re)){
  printf("buf res %s id  \n",Buffer_res );
  printf("%d int res\n",re );
  printf("datos encontados res %s e id %d \n",residuos_en_proteina[w].Nombre_res,residuos_en_proteina[w].Id_Res );
printf("energia string %s\n",ener );
printf("buf energia en float  %f \n",energy);
residuos_en_proteina[w].energias_dft=energy;
break;
}

}
con++;
}
}
}
 catch(char* pMsg) { cerr << endl << "Exception:" << pMsg << endl; }

}

void flexibility(char *file);

float std(float mean , vector<float> desv){
float standardDeviation=0;
  for(int i=0; i<desv.size(); ++i){
  float lin=  desv[i] - mean;
  float d= pow(lin, 2);
  standardDeviation =  standardDeviation+d;
}
standardDeviation=sqrt(standardDeviation/desv.size());
return standardDeviation;
}


};
