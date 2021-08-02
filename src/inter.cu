#include <iostream>
#include "clases.hpp"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

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

void Proteina::apartar_lista_atomos(int n){
cudaMallocManaged(&atomos_en_proteina,n*sizeof(Atomo));
}

void Proteina::destroy_list_obj_2(){
cudaFree(atomos_en_proteina);
}

void Proteina::clear_interface_list(){
cudaDeviceReset();
}

__global__ void Inter(Atomo* atomos,int No_atomos,float tol){

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
  float distancia;
  float r=0;

if(uniqueThread<No_atomos){

for(int i=0;i< No_atomos;i++){
if(atomos[uniqueThread].chain_id!=atomos[i].chain_id){
distancia=atomos[uniqueThread].radii+atomos[i].radii+tol;
r=sqrt(pow(atomos[uniqueThread].x-atomos[i].x,2)+pow(atomos[uniqueThread].y-atomos[i].y,2)+pow(atomos[uniqueThread].z-atomos[i].z,2));

if(r<=distancia){
atomos[uniqueThread].interfaz=true;
break;
}
}else{
//
}
}
//}
}else{
//se_paso=true;
}

}

void Proteina::interface(){
int a;
int cadena_residuo;
int count=0;
float tolerancia_interfaz=0.5;
apartar_lista_atomos(No_atomos_en_prot);
//int size_char=0;
for(int i=0; i< No_residuos;i++){
a=residuos_en_proteina[i].No_atomos;
cadena_residuo=residuos_en_proteina[i].id_cadena;
for(int j=0; j< a;j++){
residuos_en_proteina[i].atomos_en_residuo[j].cadena_identificador(cadena_residuo);
atomos_en_proteina[count]=residuos_en_proteina[i].atomos_en_residuo[j];
count++;
}
}

dim3 dimGrid(30,30);
dim3 dimBlock(32,16);
printf("cont %d \n",count );
printf("atomos %d \n",No_atomos_en_prot );

Inter<<<dimGrid,dimBlock>>>(atomos_en_proteina,No_atomos_en_prot,tolerancia_interfaz);
cudaDeviceSynchronize();
cudaCheckErrors("kernel fail");
int buffer;
int s;
for(int k=0; k < No_atomos_en_prot; ++k){
if(atomos_en_proteina[k].interfaz==true) {
buffer=atomos_en_proteina[k].Id_residuo;

for(int j=0;j<No_residuos;j++){
s=residuos_en_proteina[j].Id_Res;
if(buffer==s){
residuos_en_proteina[j].interfaz=true;
}
}
//printf("atomo de interfaz del residuo %d \n",atomos_en_proteina[k].Id_residuo);
}
}

for(int n=0;n<No_residuos;n++){
if(residuos_en_proteina[n].interfaz==true){
printf("residuo %s de interfaz ID %d en cadena %s \n",residuos_en_proteina[n].Nombre_res,residuos_en_proteina[n].Id_Res,residuos_en_proteina[n].Nombre_cad);
}
}

destroy_list_obj_2();
}
