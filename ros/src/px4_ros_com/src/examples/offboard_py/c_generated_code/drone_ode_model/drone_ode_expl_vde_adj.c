/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) drone_ode_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[35] = {31, 1, 0, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
static const casadi_int casadi_s3[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

/* drone_ode_expl_vde_adj:(i0[17],i1[17],i2[4],i3[31])->(o0[21]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a3, a4, a5, a6, a7, a8, a9;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[0]? arg[0][13] : 0;
  a1=arg[0]? arg[0][14] : 0;
  a0=(a0+a1);
  a1=arg[0]? arg[0][15] : 0;
  a0=(a0+a1);
  a1=arg[0]? arg[0][16] : 0;
  a0=(a0+a1);
  a1=arg[0]? arg[0][3] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][4] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a4=arg[0]? arg[0][5] : 0;
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a5=arg[0]? arg[0][6] : 0;
  a6=casadi_sq(a5);
  a2=(a2+a6);
  a6=(a1/a2);
  a7=arg[1]? arg[1][9] : 0;
  a8=arg[3]? arg[3][0] : 0;
  a7=(a7/a8);
  a9=(a6*a7);
  a10=(a3/a2);
  a11=arg[1]? arg[1][8] : 0;
  a11=(a11/a8);
  a12=(a10*a11);
  a9=(a9-a12);
  a12=(a4/a2);
  a13=arg[1]? arg[1][7] : 0;
  a13=(a13/a8);
  a8=(a12*a13);
  a9=(a9+a8);
  a8=(a0*a9);
  a14=(a1*a0);
  a15=(a14*a7);
  a16=(a3*a0);
  a17=(a16*a11);
  a15=(a15-a17);
  a17=(a4*a0);
  a18=(a17*a13);
  a15=(a15+a18);
  a18=(a15/a2);
  a8=(a8+a18);
  a18=(a1+a1);
  a19=(a12/a2);
  a20=(a5*a0);
  a21=(a20*a11);
  a22=(a17*a7);
  a21=(a21-a22);
  a22=(a14*a13);
  a21=(a21+a22);
  a19=(a19*a21);
  a22=(a5/a2);
  a23=(a22/a2);
  a24=(a20*a7);
  a17=(a17*a11);
  a24=(a24+a17);
  a17=(a16*a13);
  a24=(a24+a17);
  a23=(a23*a24);
  a19=(a19+a23);
  a23=(a6/a2);
  a23=(a23*a15);
  a19=(a19+a23);
  a23=(a10/a2);
  a20=(a20*a13);
  a16=(a16*a7);
  a14=(a14*a11);
  a16=(a16+a14);
  a20=(a20-a16);
  a23=(a23*a20);
  a19=(a19+a23);
  a18=(a18*a19);
  a8=(a8-a18);
  a18=arg[0]? arg[0][12] : 0;
  a23=5.0000000000000000e-01;
  a16=arg[1]? arg[1][6] : 0;
  a16=(a23*a16);
  a14=(a18*a16);
  a8=(a8+a14);
  a14=arg[0]? arg[0][11] : 0;
  a15=arg[1]? arg[1][5] : 0;
  a15=(a23*a15);
  a17=(a14*a15);
  a8=(a8+a17);
  a17=arg[0]? arg[0][10] : 0;
  a25=arg[1]? arg[1][4] : 0;
  a25=(a23*a25);
  a26=(a17*a25);
  a8=(a8+a26);
  if (res[0]!=0) res[0][3]=a8;
  a8=(a22*a13);
  a26=(a10*a7);
  a27=(a6*a11);
  a26=(a26+a27);
  a8=(a8-a26);
  a26=(a0*a8);
  a20=(a20/a2);
  a26=(a26+a20);
  a20=(a3+a3);
  a20=(a20*a19);
  a26=(a26-a20);
  a20=(a14*a16);
  a26=(a26+a20);
  a20=(a18*a15);
  a26=(a26-a20);
  a20=arg[1]? arg[1][3] : 0;
  a23=(a23*a20);
  a20=(a17*a23);
  a26=(a26-a20);
  if (res[0]!=0) res[0][4]=a26;
  a21=(a21/a2);
  a26=(a22*a11);
  a20=(a12*a7);
  a26=(a26-a20);
  a6=(a6*a13);
  a26=(a26+a6);
  a6=(a0*a26);
  a21=(a21+a6);
  a6=(a4+a4);
  a6=(a6*a19);
  a21=(a21-a6);
  a6=(a17*a16);
  a21=(a21-a6);
  a6=(a18*a25);
  a21=(a21+a6);
  a6=(a14*a23);
  a21=(a21-a6);
  if (res[0]!=0) res[0][5]=a21;
  a24=(a24/a2);
  a2=(a5+a5);
  a2=(a2*a19);
  a24=(a24-a2);
  a22=(a22*a7);
  a12=(a12*a11);
  a22=(a22+a12);
  a10=(a10*a13);
  a22=(a22+a10);
  a0=(a0*a22);
  a24=(a24+a0);
  a0=(a17*a15);
  a24=(a24+a0);
  a0=(a14*a25);
  a24=(a24-a0);
  a0=(a18*a23);
  a24=(a24-a0);
  if (res[0]!=0) res[0][6]=a24;
  a24=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][7]=a24;
  a24=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][8]=a24;
  a24=arg[1]? arg[1][2] : 0;
  if (res[0]!=0) res[0][9]=a24;
  a24=arg[3]? arg[3][4] : 0;
  a0=(a24*a18);
  a10=arg[1]? arg[1][11] : 0;
  a13=arg[3]? arg[3][3] : 0;
  a10=(a10/a13);
  a12=(a0*a10);
  a11=(a13*a14);
  a7=arg[1]? arg[1][12] : 0;
  a7=(a7/a24);
  a2=(a11*a7);
  a12=(a12-a2);
  a2=arg[3]? arg[3][2] : 0;
  a19=(a14*a7);
  a21=(a18*a10);
  a19=(a19-a21);
  a19=(a2*a19);
  a12=(a12+a19);
  a19=(a4*a16);
  a12=(a12-a19);
  a19=(a5*a15);
  a12=(a12+a19);
  a19=(a1*a25);
  a12=(a12+a19);
  a19=(a3*a23);
  a12=(a12-a19);
  if (res[0]!=0) res[0][10]=a12;
  a12=(a2*a17);
  a19=(a12*a7);
  a21=arg[1]? arg[1][10] : 0;
  a21=(a21/a2);
  a18=(a18*a21);
  a2=(a17*a7);
  a18=(a18-a2);
  a13=(a13*a18);
  a19=(a19+a13);
  a0=(a0*a21);
  a19=(a19-a0);
  a0=(a3*a16);
  a19=(a19+a0);
  a0=(a1*a15);
  a19=(a19+a0);
  a0=(a5*a25);
  a19=(a19-a0);
  a0=(a4*a23);
  a19=(a19-a0);
  if (res[0]!=0) res[0][11]=a19;
  a11=(a11*a21);
  a12=(a12*a10);
  a11=(a11-a12);
  a17=(a17*a10);
  a14=(a14*a21);
  a17=(a17-a14);
  a24=(a24*a17);
  a11=(a11+a24);
  a16=(a1*a16);
  a11=(a11+a16);
  a15=(a3*a15);
  a11=(a11-a15);
  a25=(a4*a25);
  a11=(a11+a25);
  a23=(a5*a23);
  a11=(a11-a23);
  if (res[0]!=0) res[0][12]=a11;
  a11=arg[3]? arg[3][9] : 0;
  a11=(a11*a10);
  a23=3.;
  a25=arg[1]? arg[1][13] : 0;
  a25=(a23*a25);
  a15=arg[3]? arg[3][13] : 0;
  a16=(a15*a7);
  a16=(a25+a16);
  a11=(a11-a16);
  a16=arg[3]? arg[3][5] : 0;
  a16=(a16*a21);
  a11=(a11-a16);
  a1=(a1*a9);
  a3=(a3*a8);
  a1=(a1+a3);
  a4=(a4*a26);
  a1=(a1+a4);
  a5=(a5*a22);
  a1=(a1+a5);
  a11=(a11+a1);
  if (res[0]!=0) res[0][13]=a11;
  a11=(a15*a7);
  a5=arg[1]? arg[1][14] : 0;
  a5=(a23*a5);
  a11=(a11-a5);
  a22=arg[3]? arg[3][10] : 0;
  a22=(a22*a10);
  a11=(a11-a22);
  a22=arg[3]? arg[3][6] : 0;
  a22=(a22*a21);
  a11=(a11-a22);
  a11=(a11+a1);
  if (res[0]!=0) res[0][14]=a11;
  a11=arg[3]? arg[3][7] : 0;
  a11=(a11*a21);
  a22=arg[1]? arg[1][15] : 0;
  a22=(a23*a22);
  a4=(a15*a7);
  a4=(a22+a4);
  a26=arg[3]? arg[3][11] : 0;
  a26=(a26*a10);
  a4=(a4+a26);
  a11=(a11-a4);
  a11=(a11+a1);
  if (res[0]!=0) res[0][15]=a11;
  a15=(a15*a7);
  a7=arg[1]? arg[1][16] : 0;
  a23=(a23*a7);
  a15=(a15-a23);
  a7=arg[3]? arg[3][12] : 0;
  a7=(a7*a10);
  a15=(a15+a7);
  a7=arg[3]? arg[3][8] : 0;
  a7=(a7*a21);
  a15=(a15+a7);
  a15=(a15+a1);
  if (res[0]!=0) res[0][16]=a15;
  a15=1.0000000000000001e-01;
  a25=(a15*a25);
  if (res[0]!=0) res[0][17]=a25;
  a5=(a15*a5);
  if (res[0]!=0) res[0][18]=a5;
  a22=(a15*a22);
  if (res[0]!=0) res[0][19]=a22;
  a15=(a15*a23);
  if (res[0]!=0) res[0][20]=a15;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_expl_vde_adj_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_expl_vde_adj_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_expl_vde_adj_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_expl_vde_adj_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif