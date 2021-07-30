#include <iostream>
#include "clases.hpp"
#include <math.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;

void Proteina::clear(){
cudaDeviceReset();
Lista_residuos.clear();
Lista_atomos_proteina.clear();
}

void Atomo::apartar_memoria_3(){
cudaMallocManaged(&ids,300*sizeof(int));
cudaMallocManaged(&ids2,300*sizeof(int));
cudaMallocManaged(&xs,300*sizeof(float));
cudaMallocManaged(&ys,300*sizeof(float));
cudaMallocManaged(&zs,300*sizeof(float));
cudaMallocManaged(&radd,300*sizeof(float));
//cudaMallocManaged(&ID_cajas,300*sizeof(int));
}

void Boxx::apartar_memoria_at(int n){
cudaMallocManaged(&atomos_caja,n*sizeof(Atomo));
}


void Residuo::apartar_memoria_res(int n) {
//  printf("aparto memoria \n");
cudaMallocManaged(&atomos_en_residuo,n*sizeof(Atomo));
}
//#endif

void Proteina::apartar_memoria_cajas(int n){
  cudaMallocManaged(&Cajas,n*sizeof(Boxx));
}

void Proteina::apartar_memoria_proteina(int n) {
//  #ifdef USE_CUDA
  cudaMallocManaged(&residuos_en_proteina,n*sizeof(Residuo));
  //  #endif
  }

  void Proteina::destroy_list_obj(){
cudaFree(residuos_en_proteina);
  }


void Puntos_fibo(int p, float** &lista ){
float c=M_PI*(3.0-sqrt(5.0));
float a=2.0/p;
float s;
float ss;
float r;
float f;
cudaMallocManaged(&lista, p*sizeof(float *));
for (int i = 0; i < p; ++i){
s=i*a-1+(a/2);
ss=s*s;
r=sqrt(1-ss);
f=i*c;
cudaMallocManaged(&(lista[i]), 3*sizeof(float));
  lista [i][0]=cos(f)*r;
  lista [i][1]=s;
  lista [i][2]=sin(f)*r;
}

}


__device__ float cajas_externas (int idBlock,int idThread,int ii,Boxx* Cajas,float d2){

  int  q=Cajas[ii].No_atoms;
  float xxx,yyy,zzz;
    for (int m = 0; m < q; ++m){

   xxx=(Cajas[idBlock].atomos_caja[idThread].x)-(Cajas[ii].atomos_caja[m].x);
   yyy=(Cajas[idBlock].atomos_caja[idThread].y)-(Cajas[ii].atomos_caja[m].y);
   zzz=(Cajas[idBlock].atomos_caja[idThread].z)-(Cajas[ii].atomos_caja[m].z);
  float MM=(xxx*xxx)+(yyy*yyy)+(zzz*zzz);
      if(MM<=d2){
      int mm=0;
//  while( (Cajas[idBlock].atomos_caja[idThread].id_cajas[mm]!=0) && (Cajas[idBlock].atomos_caja[idThread].cajitas[mm]!=0)  ){
      while( Cajas[idBlock].atomos_caja[idThread].ids[mm]!=0    ){
      mm++;
      }
     Cajas[idBlock].atomos_caja[idThread].ids[mm]=Cajas[ii].atomos_caja[m].No_Atomo;
      }
    }

}

__device__ float busqueda(int indice, Boxx* Cajas,int n,float& xe, float& ye, float& ze, float& rr, int& aidi,int idBlock,int idThread){

bool flag=true;
int V;
  for (int t = 0; t < n; ++t){
      V=Cajas[t].No_atoms;
    for (int u  = 0; u < V; ++u){
  if(  Cajas[t].atomos_caja[u].No_Atomo == indice  ){

xe=Cajas[t].atomos_caja[u].x;
ye=Cajas[t].atomos_caja[u].y;
ze=Cajas[t].atomos_caja[u].z;
rr=Cajas[t].atomos_caja[u].radii;
aidi=indice;
flag=false;
break;

  }
  }
 if(flag==false){
   break;
  }
  }
}

__global__ void asa_2(float** lista,Boxx* Cajas,int n,int puntos,float punta, float t){

  int Q;
  int thx= threadIdx.x;
  int thy= threadIdx.y;
  int bx= blockIdx.x;
  int by= blockIdx.y;
  //coordernadas del hilo
  int nBlocks= gridDim.x * gridDim.y;
  //per bloque
  int nThreads= blockDim.x*blockDim.y;
  int idThread= (thy*blockDim.x + thx);
  int idBlock= (by*gridDim.x + bx);
  int uniqueThread= nThreads*idBlock + idThread;
  if(idBlock < n){
    Q=Cajas[idBlock].No_atoms;
    //printf("No atomos %d \n",Q );
    if(idThread<Q){

  float pp=punta+(Cajas[idBlock].atomos_caja[idThread].radii);
  float xs;
  float ys;
  float zs;
  int N=0;
  int ad=0;
  int ident=0;
  float rap=0;
  float xxx;
  float yyy;
  float zzz;
  float dx,dy,dz,rod,rod2,D;
 bool flag;

  for (int k  = 0; k < puntos; ++k){
  ad=0;
  bool acc=true;
  xs=lista[k][0]*pp+(Cajas[idBlock].atomos_caja[idThread].x);
  ys=lista[k][1]*pp+(Cajas[idBlock].atomos_caja[idThread].y);
  zs=lista[k][2]*pp+(Cajas[idBlock].atomos_caja[idThread].z);

  for (int  a=0; a < Cajas[idBlock].atomos_caja[idThread].No_Atomos2 ; ++a){
   ad=Cajas[idBlock].atomos_caja[idThread].ids2[a];
   xxx=Cajas[idBlock].atomos_caja[idThread].xs[a];
    yyy=Cajas[idBlock].atomos_caja[idThread].ys[a];
    zzz=Cajas[idBlock].atomos_caja[idThread].zs[a];
    rap=Cajas[idBlock].atomos_caja[idThread].radd[a];
//busqueda(ad,Cajas,n,xxx, yyy, zzz,  radd, ident,idBlock,idThread);
 rod=rap+punta;
 rod2=rod*rod;
 dx=xxx-xs;
 dy=yyy-ys;
 dz=zzz-zs;
 D=(dx*dx)+(dy*dy)+(dz*dz);

if  ( Cajas[idBlock].atomos_caja[idThread].No_Atomo==1 ){
if(k==0){
//printf("D es %f y r2 es %f\n",D,rod2 );
}
}

if(D<rod2){
  acc=false;
  break;
}
}

if(acc==true){
N++;
}
}
Cajas[idBlock].atomos_caja[idThread]. area_atomo=t*N*pp*pp;
}
}

}


__global__ void asa_1(Boxx* Cajas,int n,float d_max,int puntos,float punta){

  int thx= threadIdx.x;
  int thy= threadIdx.y;
  int bx= blockIdx.x;
  int by= blockIdx.y;
  //coordernadas del hilo
  int nBlocks= gridDim.x * gridDim.y;
  //per bloque
  int nThreads= blockDim.x*blockDim.y;
  int idThread= (thy*blockDim.x + thx);
  int idBlock= (by*gridDim.x + bx);
  int uniqueThread= nThreads*idBlock + idThread;


float d2=d_max*d_max;
int Q;
if(idBlock < n){
Q=Cajas[idBlock].No_atoms;
  //printf("No atomos %d \n",Q );
if(idThread<Q){
if(   (Cajas[idBlock].x==-3)  &&  (Cajas[idBlock].y==9) &&  (Cajas[idBlock].z==4)    ){
//  printf("atomos  Cajas x  %d  \n",  Cajas[idBlock].atomos_caja[idThread].No_Atomo);
}
/////////////////////////////////////////////////////////////////////////////////////
  //CAJAS_INTERNAS
float xx,yy,zz;

for (int i = 0; i < Q; ++i){
xx=(Cajas[idBlock].atomos_caja[idThread].x)-(Cajas[idBlock].atomos_caja[i].x);
yy=(Cajas[idBlock].atomos_caja[idThread].y)-(Cajas[idBlock].atomos_caja[i].y);
zz=(Cajas[idBlock].atomos_caja[idThread].z)-(Cajas[idBlock].atomos_caja[i].z);
float M=(xx*xx)+(yy*yy)+(zz*zz);
//float d2=d_max*d_max;
//printf("M float %f \n",M );
if(M<=d2){
int j=0;
while( Cajas[idBlock].atomos_caja[idThread].ids[j]!=0  ){
j++;
}
Cajas[idBlock].atomos_caja[idThread].ids[j]=Cajas[idBlock].atomos_caja[i].No_Atomo;
}
}
//CAJAS EXTERNAS
for (int ii  = 0; ii < n; ++ii){
  int cx=Cajas[ii].x;
  int cy=Cajas[ii].y;
  int cz=Cajas[ii].z;

  if ( ((Cajas[idBlock].x-1)==cx ) && ((Cajas[idBlock].y + 1)==cy) && ((Cajas[idBlock].z)==cz) ){
cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x)==cx ) && ((Cajas[idBlock].y + 1)==cy) && ( (Cajas[idBlock].z)==cz)  ){
    cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x+1)==cx ) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z)==cz)   ){
    cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z)==cz) ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z)==cz) ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z)==cz) ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z)==cz)  ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z)==cz)  ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

///////////////////////////////////////////////////////////////////////////////
  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z +1)==cz)  ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z +1)==cz)  ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z +1)==cz) ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z +1)==cz) ){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z +1)==cz)  ){
    cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z +1)==cz) ){
    cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z +1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z +1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z +1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
    /////////////////////////////////////////////////////////////////////////
  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y+1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x-1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }
  if ( ((Cajas[idBlock].x+1)==cx) && ((Cajas[idBlock].y-1)==cy) && ((Cajas[idBlock].z-1)==cz)){
  cajas_externas (idBlock,idThread ,ii,Cajas,d2);
  }

}
////////////////////////////////////////////////////////////
//BENCHMARK
int b=0;
if(idBlock==0){
while(  Cajas[idBlock].atomos_caja[idThread].ids[b]  !=0  ){
b++;
}
}

if((Cajas[idBlock].x==-3)  &&  (Cajas[idBlock].y==9) &&  (Cajas[idBlock].z==4) ){
if  ( Cajas[idBlock].atomos_caja[idThread].No_Atomo==1 ){
//  printf("Id atomo caja   %d  \n",  Cajas[idBlock].atomos_caja[idThread].No_Atomo);
int fer=0;
while(  Cajas[idBlock].atomos_caja[0].ids[fer] !=0  ){
//printf("%d id de atomos en caja 0 atomo 0/1  \n",Cajas[idBlock].atomos_caja[0].ids[fer] );
fer++;
;
}

}
}
//////////////////////////////////////////////////////////////////////////
int iit=0;
float dis;
float difx;
float dify;
float difz;
float g;
int indice;
float rad=Cajas[idBlock].atomos_caja[idThread].radii+(2.0*punta);
int V;
float xe,ye,ze,rr;
int aidi;
int cuenta=0;
int h=0;

while(Cajas[idBlock].atomos_caja[idThread].ids[iit]!=0 ){

indice=Cajas[idBlock].atomos_caja[idThread].ids[iit];
 busqueda(indice,Cajas,n,xe, ye, ze,  rr, aidi,idBlock,idThread);

if (aidi!=Cajas[idBlock].atomos_caja[idThread]. No_Atomo) {
//rad
difx= (Cajas[idBlock].atomos_caja[idThread].x)-xe;
dify= (Cajas[idBlock].atomos_caja[idThread].y)-ye;
difz= (Cajas[idBlock].atomos_caja[idThread].z)-ze;
dis= difx*difx+dify*dify+difz*difz;
g=(rad+rr)*(rad+rr);
if(dis< g ){

while( Cajas[idBlock].atomos_caja[idThread].ids2[h]!=0  ){
//  while(   Cajas[idBlock].atomos_caja[idThread].ids[h]!=0      ){
  h++;
  }
Cajas[idBlock].atomos_caja[idThread].ids2[h]=aidi;
Cajas[idBlock].atomos_caja[idThread].xs[h]=xe;
Cajas[idBlock].atomos_caja[idThread].ys[h]=ye;
Cajas[idBlock].atomos_caja[idThread].zs[h]=ze;
Cajas[idBlock].atomos_caja[idThread].radd[h]=rr;
cuenta++;
h=cuenta;
}
}
iit++;
}
Cajas[idBlock].atomos_caja[idThread].No_Atomos2=cuenta;
////////////////////////////////////////////////////////////////////////////////////
//BENCHMARK
int ff=0;
if( (Cajas[idBlock].x==-3)  &&  (Cajas[idBlock].y==9) &&  (Cajas[idBlock].z==4)){
if  ( Cajas[idBlock].atomos_caja[idThread].No_Atomo==1 ){
//printf("Id atomo caja   %d  \n",  Cajas[idBlock].atomos_caja[idThread].No_Atomo);
while(  Cajas[idBlock].atomos_caja[0].ids2[ff] !=0  ){
//printf("%d id de atomos en caja 0 atomo 0/2  \n",Cajas[idBlock].atomos_caja[0].ids2[ff] );
ff++;
}
//printf("%d cuantos hay 2 \n", ff);
}
}

}
}
}

void Proteina::asa(float punta,float& s,int puntos){

printf("Usado %d puntos \n",puntos );
float **lista_puntos;
Puntos_fibo(puntos,lista_puntos);
float t= (4.0*M_PI)/puntos;
//printf("%f un ejemplo de punto fibo aparte fuera \n",lista_puntos[0][0] );
  int max=0;
  int p=0;
  //printf("%d y b size es \n", Lista_Cajas.size() );
  for (int i = 0; i <  Lista_Cajas.size(); ++i){
    p= Lista_Cajas[i].Lista_atomos_caja.size();

    if(max<p){
    max=p;
    }
    Lista_Cajas[i].Add_No(p);
    Lista_Cajas[i].apartar_memoria_at(p);
  }

int c;
for (int j = 0; j < Lista_Cajas.size(); ++j){
c=Lista_Cajas[j].Lista_atomos_caja.size();
for (int jj = 0; jj < c; ++jj){
 Lista_Cajas[j].atomos_caja[jj]=Lista_Cajas[j].Lista_atomos_caja[jj];
 Lista_Cajas[j].atomos_caja[jj].apartar_memoria_3();
}
}
//printf("%d numero atomos caja 5 \n",Lista_Cajas[5]. No_atoms );
apartar_memoria_cajas(Lista_Cajas.size());
for (int k = 0; k < Lista_Cajas.size(); ++k){
Cajas[k]=Lista_Cajas[k];
}
printf("%d longitud vec cajas \n",Lista_Cajas.size());

dim3 dimGrid(30,30);
dim3 dimBlock(32,16);              // Num. de bloques en (x,y) por grid
  //  dim3 dimBlock(16,8);
printf("dmax %f \n",d_max );
printf("%d tamano n cajas \n", Lista_Cajas.size());
asa_1<<<dimGrid,dimBlock>>>( Cajas, Lista_Cajas.size() ,d_max ,puntos ,punta);
cudaDeviceSynchronize();
cudaCheckErrors("kernel fail");
asa_2<<<dimGrid,dimBlock>>>(lista_puntos, Cajas, Lista_Cajas.size()  ,puntos ,punta,t);
cudaDeviceSynchronize();
cudaCheckErrors("kernel fail");
float suma=0;
int po=0;
int ress=0;
int At1=0;
int identificador=0;
for (int k=0; k <  Lista_Cajas.size(); ++k){
po= Cajas[k].No_atoms;
for (int kk = 0; kk < po; ++kk){
suma=(Cajas[k].atomos_caja[kk].area_atomo)+suma;
ress=Cajas[k].atomos_caja[kk].Id_residuo;
At1=Cajas[k].atomos_caja[kk].No_Atomos2;
identificador=Cajas[k].atomos_caja[kk].No_Atomo;
//    printf("AT1 %d \n",At1 );
for(int kkk=0; kkk< No_residuos;++kkk){
if(  residuos_en_proteina[kkk].Id_Res==ress){
residuos_en_proteina[kkk].area_res= (residuos_en_proteina[kkk].area_res)+(Cajas[k].atomos_caja[kk].area_atomo);
residuos_en_proteina[kkk].contactos1=residuos_en_proteina[kkk].contactos1+At1;
//  printf("%f areas aqui \n",Cajas[k].atomos_caja[kk].area_atomo );
    break;
    }
    }
}
}
s=suma;
printf("%.3f suma \n",suma );
destroy_list_obj();
}
