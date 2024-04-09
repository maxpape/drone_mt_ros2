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
  #define CASADI_PREFIX(ID) drone_ode_cost_ext_cost_0_fun_jac_hess_ ## ID
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
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
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

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s6[34] = {11, 11, 0, 1, 2, 3, 4, 8, 12, 16, 20, 20, 20, 20, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
static const casadi_int casadi_s7[14] = {0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[7],i1[4],i2[],i3[21])->(o0,o1[11],o2[11x11,20nz],o3[],o4[0x11]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a9;
  a0=2.;
  a1=arg[3]? arg[3][14] : 0;
  a2=arg[0]? arg[0][0] : 0;
  a3=casadi_sq(a2);
  a4=arg[0]? arg[0][1] : 0;
  a5=casadi_sq(a4);
  a3=(a3+a5);
  a5=arg[0]? arg[0][2] : 0;
  a6=casadi_sq(a5);
  a3=(a3+a6);
  a6=arg[0]? arg[0][3] : 0;
  a7=casadi_sq(a6);
  a3=(a3+a7);
  a7=(a2/a3);
  a8=(a1*a7);
  a9=arg[3]? arg[3][15] : 0;
  a10=(a4/a3);
  a11=(a9*a10);
  a12=arg[3]? arg[3][16] : 0;
  a13=(a5/a3);
  a14=(a12*a13);
  a11=(a11+a14);
  a14=arg[3]? arg[3][17] : 0;
  a15=(a6/a3);
  a16=(a14*a15);
  a11=(a11+a16);
  a8=(a8+a11);
  a8=(a0*a8);
  a11=casadi_sq(a2);
  a16=casadi_sq(a4);
  a11=(a11+a16);
  a16=casadi_sq(a5);
  a11=(a11+a16);
  a16=casadi_sq(a6);
  a11=(a11+a16);
  a16=(a2/a11);
  a17=(a1*a16);
  a18=(a4/a11);
  a19=(a9*a18);
  a20=(a5/a11);
  a21=(a12*a20);
  a19=(a19+a21);
  a21=(a6/a11);
  a22=(a14*a21);
  a19=(a19+a22);
  a17=(a17+a19);
  a19=(a8*a17);
  a22=(a7*a9);
  a23=(a1*a10);
  a22=(a22-a23);
  a23=(a14*a13);
  a24=(a12*a15);
  a23=(a23-a24);
  a22=(a22+a23);
  a22=(a0*a22);
  a23=(a16*a9);
  a24=(a1*a18);
  a23=(a23-a24);
  a24=(a14*a20);
  a25=(a12*a21);
  a24=(a24-a25);
  a23=(a23+a24);
  a24=(a22*a23);
  a19=(a19+a24);
  a24=(a7*a12);
  a25=(a1*a13);
  a24=(a24-a25);
  a25=(a9*a15);
  a26=(a14*a10);
  a25=(a25-a26);
  a24=(a24+a25);
  a24=(a0*a24);
  a25=(a16*a12);
  a26=(a1*a20);
  a25=(a25-a26);
  a26=(a9*a21);
  a27=(a14*a18);
  a26=(a26-a27);
  a25=(a25+a26);
  a26=(a24*a25);
  a19=(a19+a26);
  a26=(a7*a14);
  a27=(a1*a15);
  a26=(a26-a27);
  a27=(a12*a10);
  a28=(a9*a13);
  a27=(a27-a28);
  a26=(a26+a27);
  a26=(a0*a26);
  a27=(a16*a14);
  a28=(a1*a21);
  a27=(a27-a28);
  a28=(a12*a18);
  a29=(a9*a20);
  a28=(a28-a29);
  a27=(a27+a28);
  a28=(a26*a27);
  a19=(a19+a28);
  a28=arg[1]? arg[1][0] : 0;
  a29=casadi_sq(a28);
  a30=arg[1]? arg[1][1] : 0;
  a31=casadi_sq(a30);
  a29=(a29+a31);
  a31=arg[1]? arg[1][2] : 0;
  a32=casadi_sq(a31);
  a29=(a29+a32);
  a32=arg[1]? arg[1][3] : 0;
  a33=casadi_sq(a32);
  a29=(a29+a33);
  a19=(a19+a29);
  if (res[0]!=0) res[0][0]=a19;
  a28=(a28+a28);
  if (res[1]!=0) res[1][0]=a28;
  a30=(a30+a30);
  if (res[1]!=0) res[1][1]=a30;
  a31=(a31+a31);
  if (res[1]!=0) res[1][2]=a31;
  a32=(a32+a32);
  if (res[1]!=0) res[1][3]=a32;
  a32=(a14*a26);
  a31=(a12*a24);
  a32=(a32+a31);
  a31=(a9*a22);
  a32=(a32+a31);
  a31=(a1*a8);
  a32=(a32+a31);
  a31=(a32/a11);
  a30=(a2+a2);
  a28=(a21/a11);
  a19=(a9*a24);
  a29=(a1*a26);
  a19=(a19-a29);
  a29=(a12*a22);
  a19=(a19-a29);
  a29=(a14*a8);
  a19=(a19+a29);
  a29=(a28*a19);
  a33=(a20/a11);
  a34=(a14*a22);
  a35=(a9*a26);
  a36=(a1*a24);
  a35=(a35+a36);
  a34=(a34-a35);
  a35=(a12*a8);
  a34=(a34+a35);
  a35=(a33*a34);
  a29=(a29+a35);
  a35=(a18/a11);
  a26=(a12*a26);
  a24=(a14*a24);
  a26=(a26-a24);
  a22=(a1*a22);
  a26=(a26-a22);
  a8=(a9*a8);
  a26=(a26+a8);
  a8=(a35*a26);
  a29=(a29+a8);
  a8=(a16/a11);
  a22=(a8*a32);
  a29=(a29+a22);
  a22=(a30*a29);
  a22=(a31-a22);
  a27=(a0*a27);
  a24=(a14*a27);
  a25=(a0*a25);
  a36=(a12*a25);
  a24=(a24+a36);
  a23=(a0*a23);
  a36=(a9*a23);
  a24=(a24+a36);
  a17=(a0*a17);
  a36=(a1*a17);
  a24=(a24+a36);
  a36=(a24/a3);
  a22=(a22+a36);
  a37=(a2+a2);
  a38=(a15/a3);
  a39=(a9*a25);
  a40=(a1*a27);
  a39=(a39-a40);
  a40=(a12*a23);
  a39=(a39-a40);
  a40=(a14*a17);
  a39=(a39+a40);
  a40=(a38*a39);
  a41=(a13/a3);
  a42=(a14*a23);
  a43=(a9*a27);
  a44=(a1*a25);
  a43=(a43+a44);
  a42=(a42-a43);
  a43=(a12*a17);
  a42=(a42+a43);
  a43=(a41*a42);
  a40=(a40+a43);
  a43=(a10/a3);
  a27=(a12*a27);
  a25=(a14*a25);
  a27=(a27-a25);
  a23=(a1*a23);
  a27=(a27-a23);
  a17=(a9*a17);
  a27=(a27+a17);
  a17=(a43*a27);
  a40=(a40+a17);
  a17=(a7/a3);
  a23=(a17*a24);
  a40=(a40+a23);
  a23=(a37*a40);
  a22=(a22-a23);
  if (res[1]!=0) res[1][4]=a22;
  a22=(a26/a11);
  a23=(a4+a4);
  a25=(a23*a29);
  a25=(a22-a25);
  a44=(a27/a3);
  a25=(a25+a44);
  a45=(a4+a4);
  a46=(a45*a40);
  a25=(a25-a46);
  if (res[1]!=0) res[1][5]=a25;
  a25=(a34/a11);
  a46=(a5+a5);
  a47=(a46*a29);
  a47=(a25-a47);
  a48=(a42/a3);
  a47=(a47+a48);
  a49=(a5+a5);
  a50=(a49*a40);
  a47=(a47-a50);
  if (res[1]!=0) res[1][6]=a47;
  a47=(a19/a11);
  a50=(a6+a6);
  a51=(a50*a29);
  a51=(a47-a51);
  a52=(a39/a3);
  a51=(a51+a52);
  a53=(a6+a6);
  a54=(a53*a40);
  a51=(a51-a54);
  if (res[1]!=0) res[1][7]=a51;
  a51=0.;
  if (res[1]!=0) res[1][8]=a51;
  if (res[1]!=0) res[1][9]=a51;
  if (res[1]!=0) res[1][10]=a51;
  if (res[2]!=0) res[2][0]=a0;
  if (res[2]!=0) res[2][1]=a0;
  if (res[2]!=0) res[2][2]=a0;
  if (res[2]!=0) res[2][3]=a0;
  a51=(1./a3);
  a7=(a7/a3);
  a54=(a2+a2);
  a55=(a7*a54);
  a51=(a51-a55);
  a55=(a14*a51);
  a15=(a15/a3);
  a56=(a15*a54);
  a57=(a1*a56);
  a55=(a55+a57);
  a13=(a13/a3);
  a57=(a13*a54);
  a58=(a9*a57);
  a10=(a10/a3);
  a59=(a10*a54);
  a60=(a12*a59);
  a58=(a58-a60);
  a55=(a55+a58);
  a55=(a0*a55);
  a58=(a14*a55);
  a60=(a12*a51);
  a61=(a1*a57);
  a60=(a60+a61);
  a61=(a14*a59);
  a62=(a9*a56);
  a61=(a61-a62);
  a60=(a60+a61);
  a60=(a0*a60);
  a61=(a12*a60);
  a58=(a58+a61);
  a61=(a9*a51);
  a62=(a1*a59);
  a61=(a61+a62);
  a62=(a12*a56);
  a63=(a14*a57);
  a62=(a62-a63);
  a61=(a61+a62);
  a61=(a0*a61);
  a62=(a9*a61);
  a58=(a58+a62);
  a62=(a1*a51);
  a63=(a9*a59);
  a64=(a12*a57);
  a63=(a63+a64);
  a64=(a14*a56);
  a63=(a63+a64);
  a62=(a62-a63);
  a62=(a0*a62);
  a63=(a1*a62);
  a58=(a58+a63);
  a63=(a58/a11);
  a31=(a31/a11);
  a2=(a2+a2);
  a64=(a31*a2);
  a63=(a63-a64);
  a64=(a0*a29);
  a65=(a9*a60);
  a66=(a1*a55);
  a65=(a65-a66);
  a66=(a12*a61);
  a65=(a65-a66);
  a66=(a14*a62);
  a65=(a65+a66);
  a65=(a28*a65);
  a21=(a21/a11);
  a66=(a21*a2);
  a67=(a66/a11);
  a68=(a28/a11);
  a69=(a68*a2);
  a67=(a67+a69);
  a67=(a19*a67);
  a65=(a65-a67);
  a67=(a14*a61);
  a69=(a9*a55);
  a70=(a1*a60);
  a69=(a69+a70);
  a67=(a67-a69);
  a69=(a12*a62);
  a67=(a67+a69);
  a67=(a33*a67);
  a20=(a20/a11);
  a69=(a20*a2);
  a70=(a69/a11);
  a71=(a33/a11);
  a72=(a71*a2);
  a70=(a70+a72);
  a70=(a34*a70);
  a67=(a67-a70);
  a65=(a65+a67);
  a55=(a12*a55);
  a60=(a14*a60);
  a55=(a55-a60);
  a61=(a1*a61);
  a55=(a55-a61);
  a62=(a9*a62);
  a55=(a55+a62);
  a55=(a35*a55);
  a18=(a18/a11);
  a62=(a18*a2);
  a61=(a62/a11);
  a60=(a35/a11);
  a67=(a60*a2);
  a61=(a61+a67);
  a61=(a26*a61);
  a55=(a55-a61);
  a65=(a65+a55);
  a55=(1./a11);
  a16=(a16/a11);
  a61=(a16*a2);
  a55=(a55-a61);
  a61=(a55/a11);
  a67=(a8/a11);
  a2=(a67*a2);
  a61=(a61-a2);
  a61=(a32*a61);
  a58=(a8*a58);
  a61=(a61+a58);
  a65=(a65+a61);
  a65=(a30*a65);
  a64=(a64+a65);
  a63=(a63-a64);
  a64=(a14*a55);
  a65=(a1*a66);
  a64=(a64+a65);
  a65=(a9*a69);
  a61=(a12*a62);
  a65=(a65-a61);
  a64=(a64+a65);
  a64=(a0*a64);
  a65=(a14*a64);
  a61=(a12*a55);
  a58=(a1*a69);
  a61=(a61+a58);
  a58=(a14*a62);
  a2=(a9*a66);
  a58=(a58-a2);
  a61=(a61+a58);
  a61=(a0*a61);
  a58=(a12*a61);
  a65=(a65+a58);
  a58=(a9*a55);
  a2=(a1*a62);
  a58=(a58+a2);
  a2=(a12*a66);
  a70=(a14*a69);
  a2=(a2-a70);
  a58=(a58+a2);
  a58=(a0*a58);
  a2=(a9*a58);
  a65=(a65+a2);
  a55=(a1*a55);
  a62=(a9*a62);
  a69=(a12*a69);
  a62=(a62+a69);
  a66=(a14*a66);
  a62=(a62+a66);
  a55=(a55-a62);
  a55=(a0*a55);
  a62=(a1*a55);
  a65=(a65+a62);
  a62=(a65/a3);
  a36=(a36/a3);
  a66=(a36*a54);
  a62=(a62-a66);
  a63=(a63+a62);
  a62=(a0*a40);
  a66=(a9*a61);
  a69=(a1*a64);
  a66=(a66-a69);
  a69=(a12*a58);
  a66=(a66-a69);
  a69=(a14*a55);
  a66=(a66+a69);
  a66=(a38*a66);
  a56=(a56/a3);
  a69=(a38/a3);
  a2=(a69*a54);
  a56=(a56+a2);
  a56=(a39*a56);
  a66=(a66-a56);
  a56=(a14*a58);
  a2=(a9*a64);
  a70=(a1*a61);
  a2=(a2+a70);
  a56=(a56-a2);
  a2=(a12*a55);
  a56=(a56+a2);
  a56=(a41*a56);
  a57=(a57/a3);
  a2=(a41/a3);
  a70=(a2*a54);
  a57=(a57+a70);
  a57=(a42*a57);
  a56=(a56-a57);
  a66=(a66+a56);
  a64=(a12*a64);
  a61=(a14*a61);
  a64=(a64-a61);
  a58=(a1*a58);
  a64=(a64-a58);
  a55=(a9*a55);
  a64=(a64+a55);
  a64=(a43*a64);
  a59=(a59/a3);
  a55=(a43/a3);
  a58=(a55*a54);
  a59=(a59+a58);
  a59=(a27*a59);
  a64=(a64-a59);
  a66=(a66+a64);
  a51=(a51/a3);
  a64=(a17/a3);
  a54=(a64*a54);
  a51=(a51-a54);
  a51=(a24*a51);
  a65=(a17*a65);
  a51=(a51+a65);
  a66=(a66+a51);
  a66=(a37*a66);
  a62=(a62+a66);
  a63=(a63-a62);
  if (res[2]!=0) res[2][4]=a63;
  a63=(a4+a4);
  a62=(a15*a63);
  a66=(a1*a62);
  a51=(a7*a63);
  a65=(a14*a51);
  a66=(a66-a65);
  a65=(1./a3);
  a54=(a10*a63);
  a65=(a65-a54);
  a54=(a12*a65);
  a59=(a13*a63);
  a58=(a9*a59);
  a54=(a54+a58);
  a66=(a66+a54);
  a66=(a0*a66);
  a54=(a14*a66);
  a58=(a1*a59);
  a61=(a12*a51);
  a58=(a58-a61);
  a61=(a9*a62);
  a56=(a14*a65);
  a61=(a61+a56);
  a58=(a58-a61);
  a58=(a0*a58);
  a61=(a12*a58);
  a54=(a54+a61);
  a61=(a12*a62);
  a56=(a14*a59);
  a61=(a61-a56);
  a56=(a9*a51);
  a57=(a1*a65);
  a56=(a56+a57);
  a61=(a61-a56);
  a61=(a0*a61);
  a56=(a9*a61);
  a54=(a54+a56);
  a56=(a9*a65);
  a57=(a12*a59);
  a56=(a56-a57);
  a57=(a14*a62);
  a56=(a56-a57);
  a57=(a1*a51);
  a56=(a56-a57);
  a56=(a0*a56);
  a57=(a1*a56);
  a54=(a54+a57);
  a57=(a54/a11);
  a4=(a4+a4);
  a70=(a31*a4);
  a57=(a57-a70);
  a70=(a9*a58);
  a72=(a1*a66);
  a70=(a70-a72);
  a72=(a12*a61);
  a70=(a70-a72);
  a72=(a14*a56);
  a70=(a70+a72);
  a70=(a28*a70);
  a72=(a21*a4);
  a73=(a72/a11);
  a74=(a68*a4);
  a73=(a73+a74);
  a73=(a19*a73);
  a70=(a70-a73);
  a73=(a14*a61);
  a74=(a9*a66);
  a75=(a1*a58);
  a74=(a74+a75);
  a73=(a73-a74);
  a74=(a12*a56);
  a73=(a73+a74);
  a73=(a33*a73);
  a74=(a20*a4);
  a75=(a74/a11);
  a76=(a71*a4);
  a75=(a75+a76);
  a75=(a34*a75);
  a73=(a73-a75);
  a70=(a70+a73);
  a73=(1./a11);
  a75=(a18*a4);
  a73=(a73-a75);
  a75=(a73/a11);
  a76=(a60*a4);
  a75=(a75-a76);
  a75=(a26*a75);
  a66=(a12*a66);
  a58=(a14*a58);
  a66=(a66-a58);
  a61=(a1*a61);
  a66=(a66-a61);
  a56=(a9*a56);
  a66=(a66+a56);
  a56=(a35*a66);
  a75=(a75+a56);
  a70=(a70+a75);
  a54=(a8*a54);
  a75=(a16*a4);
  a56=(a75/a11);
  a61=(a67*a4);
  a56=(a56+a61);
  a56=(a32*a56);
  a54=(a54-a56);
  a70=(a70+a54);
  a54=(a30*a70);
  a57=(a57-a54);
  a54=(a1*a72);
  a56=(a14*a75);
  a54=(a54-a56);
  a56=(a12*a73);
  a61=(a9*a74);
  a56=(a56+a61);
  a54=(a54+a56);
  a54=(a0*a54);
  a56=(a14*a54);
  a61=(a1*a74);
  a58=(a12*a75);
  a61=(a61-a58);
  a58=(a9*a72);
  a76=(a14*a73);
  a58=(a58+a76);
  a61=(a61-a58);
  a61=(a0*a61);
  a58=(a12*a61);
  a56=(a56+a58);
  a58=(a12*a72);
  a76=(a14*a74);
  a58=(a58-a76);
  a76=(a9*a75);
  a77=(a1*a73);
  a76=(a76+a77);
  a58=(a58-a76);
  a58=(a0*a58);
  a76=(a9*a58);
  a56=(a56+a76);
  a73=(a9*a73);
  a74=(a12*a74);
  a73=(a73-a74);
  a72=(a14*a72);
  a73=(a73-a72);
  a75=(a1*a75);
  a73=(a73-a75);
  a73=(a0*a73);
  a75=(a1*a73);
  a56=(a56+a75);
  a75=(a56/a3);
  a72=(a36*a63);
  a75=(a75-a72);
  a57=(a57+a75);
  a75=(a9*a61);
  a72=(a1*a54);
  a75=(a75-a72);
  a72=(a12*a58);
  a75=(a75-a72);
  a72=(a14*a73);
  a75=(a75+a72);
  a75=(a38*a75);
  a62=(a62/a3);
  a72=(a69*a63);
  a62=(a62+a72);
  a62=(a39*a62);
  a75=(a75-a62);
  a62=(a14*a58);
  a72=(a9*a54);
  a74=(a1*a61);
  a72=(a72+a74);
  a62=(a62-a72);
  a72=(a12*a73);
  a62=(a62+a72);
  a62=(a41*a62);
  a59=(a59/a3);
  a72=(a2*a63);
  a59=(a59+a72);
  a59=(a42*a59);
  a62=(a62-a59);
  a75=(a75+a62);
  a65=(a65/a3);
  a62=(a55*a63);
  a65=(a65-a62);
  a65=(a27*a65);
  a54=(a12*a54);
  a61=(a14*a61);
  a54=(a54-a61);
  a58=(a1*a58);
  a54=(a54-a58);
  a73=(a9*a73);
  a54=(a54+a73);
  a73=(a43*a54);
  a65=(a65+a73);
  a75=(a75+a65);
  a56=(a17*a56);
  a51=(a51/a3);
  a65=(a64*a63);
  a51=(a51+a65);
  a51=(a24*a51);
  a56=(a56-a51);
  a75=(a75+a56);
  a56=(a37*a75);
  a57=(a57-a56);
  if (res[2]!=0) res[2][5]=a57;
  a56=(a5+a5);
  a51=(a15*a56);
  a65=(a1*a51);
  a73=(a7*a56);
  a58=(a14*a73);
  a65=(a65-a58);
  a58=(a10*a56);
  a61=(a12*a58);
  a62=(1./a3);
  a59=(a13*a56);
  a62=(a62-a59);
  a59=(a9*a62);
  a61=(a61+a59);
  a65=(a65-a61);
  a65=(a0*a65);
  a61=(a14*a65);
  a59=(a14*a58);
  a72=(a9*a51);
  a59=(a59-a72);
  a72=(a12*a73);
  a74=(a1*a62);
  a72=(a72+a74);
  a59=(a59-a72);
  a59=(a0*a59);
  a72=(a12*a59);
  a61=(a61+a72);
  a72=(a1*a58);
  a74=(a9*a73);
  a72=(a72-a74);
  a74=(a14*a62);
  a76=(a12*a51);
  a74=(a74+a76);
  a72=(a72+a74);
  a72=(a0*a72);
  a74=(a9*a72);
  a61=(a61+a74);
  a74=(a12*a62);
  a76=(a9*a58);
  a74=(a74-a76);
  a76=(a14*a51);
  a74=(a74-a76);
  a76=(a1*a73);
  a74=(a74-a76);
  a74=(a0*a74);
  a76=(a1*a74);
  a61=(a61+a76);
  a76=(a61/a11);
  a5=(a5+a5);
  a77=(a31*a5);
  a76=(a76-a77);
  a77=(a9*a59);
  a78=(a1*a65);
  a77=(a77-a78);
  a78=(a12*a72);
  a77=(a77-a78);
  a78=(a14*a74);
  a77=(a77+a78);
  a77=(a28*a77);
  a78=(a21*a5);
  a79=(a78/a11);
  a80=(a68*a5);
  a79=(a79+a80);
  a79=(a19*a79);
  a77=(a77-a79);
  a79=(1./a11);
  a80=(a20*a5);
  a79=(a79-a80);
  a80=(a79/a11);
  a81=(a71*a5);
  a80=(a80-a81);
  a80=(a34*a80);
  a81=(a14*a72);
  a82=(a9*a65);
  a83=(a1*a59);
  a82=(a82+a83);
  a81=(a81-a82);
  a82=(a12*a74);
  a81=(a81+a82);
  a82=(a33*a81);
  a80=(a80+a82);
  a77=(a77+a80);
  a65=(a12*a65);
  a59=(a14*a59);
  a65=(a65-a59);
  a72=(a1*a72);
  a65=(a65-a72);
  a74=(a9*a74);
  a65=(a65+a74);
  a74=(a35*a65);
  a72=(a18*a5);
  a59=(a72/a11);
  a80=(a60*a5);
  a59=(a59+a80);
  a59=(a26*a59);
  a74=(a74-a59);
  a77=(a77+a74);
  a61=(a8*a61);
  a74=(a16*a5);
  a59=(a74/a11);
  a80=(a67*a5);
  a59=(a59+a80);
  a59=(a32*a59);
  a61=(a61-a59);
  a77=(a77+a61);
  a61=(a30*a77);
  a76=(a76-a61);
  a61=(a1*a78);
  a59=(a14*a74);
  a61=(a61-a59);
  a59=(a12*a72);
  a80=(a9*a79);
  a59=(a59+a80);
  a61=(a61-a59);
  a61=(a0*a61);
  a59=(a14*a61);
  a80=(a14*a72);
  a82=(a9*a78);
  a80=(a80-a82);
  a82=(a12*a74);
  a83=(a1*a79);
  a82=(a82+a83);
  a80=(a80-a82);
  a80=(a0*a80);
  a82=(a12*a80);
  a59=(a59+a82);
  a82=(a1*a72);
  a83=(a9*a74);
  a82=(a82-a83);
  a83=(a14*a79);
  a84=(a12*a78);
  a83=(a83+a84);
  a82=(a82+a83);
  a82=(a0*a82);
  a83=(a9*a82);
  a59=(a59+a83);
  a79=(a12*a79);
  a72=(a9*a72);
  a79=(a79-a72);
  a78=(a14*a78);
  a79=(a79-a78);
  a74=(a1*a74);
  a79=(a79-a74);
  a79=(a0*a79);
  a74=(a1*a79);
  a59=(a59+a74);
  a74=(a59/a3);
  a78=(a36*a56);
  a74=(a74-a78);
  a76=(a76+a74);
  a74=(a9*a80);
  a78=(a1*a61);
  a74=(a74-a78);
  a78=(a12*a82);
  a74=(a74-a78);
  a78=(a14*a79);
  a74=(a74+a78);
  a74=(a38*a74);
  a51=(a51/a3);
  a78=(a69*a56);
  a51=(a51+a78);
  a51=(a39*a51);
  a74=(a74-a51);
  a62=(a62/a3);
  a51=(a2*a56);
  a62=(a62-a51);
  a62=(a42*a62);
  a51=(a14*a82);
  a78=(a9*a61);
  a72=(a1*a80);
  a78=(a78+a72);
  a51=(a51-a78);
  a78=(a12*a79);
  a51=(a51+a78);
  a78=(a41*a51);
  a62=(a62+a78);
  a74=(a74+a62);
  a61=(a12*a61);
  a80=(a14*a80);
  a61=(a61-a80);
  a82=(a1*a82);
  a61=(a61-a82);
  a79=(a9*a79);
  a61=(a61+a79);
  a79=(a43*a61);
  a58=(a58/a3);
  a82=(a55*a56);
  a58=(a58+a82);
  a58=(a27*a58);
  a79=(a79-a58);
  a74=(a74+a79);
  a59=(a17*a59);
  a73=(a73/a3);
  a79=(a64*a56);
  a73=(a73+a79);
  a73=(a24*a73);
  a59=(a59-a73);
  a74=(a74+a59);
  a59=(a37*a74);
  a76=(a76-a59);
  if (res[2]!=0) res[2][6]=a76;
  a59=(a6+a6);
  a13=(a13*a59);
  a73=(a9*a13);
  a10=(a10*a59);
  a79=(a12*a10);
  a73=(a73-a79);
  a7=(a7*a59);
  a79=(a14*a7);
  a58=(1./a3);
  a15=(a15*a59);
  a58=(a58-a15);
  a15=(a1*a58);
  a79=(a79+a15);
  a73=(a73-a79);
  a73=(a0*a73);
  a79=(a14*a73);
  a15=(a1*a13);
  a82=(a12*a7);
  a15=(a15-a82);
  a82=(a9*a58);
  a80=(a14*a10);
  a82=(a82+a80);
  a15=(a15+a82);
  a15=(a0*a15);
  a82=(a12*a15);
  a79=(a79+a82);
  a82=(a1*a10);
  a80=(a9*a7);
  a82=(a82-a80);
  a80=(a14*a13);
  a62=(a12*a58);
  a80=(a80+a62);
  a82=(a82-a80);
  a82=(a0*a82);
  a80=(a9*a82);
  a79=(a79+a80);
  a80=(a14*a58);
  a62=(a9*a10);
  a78=(a12*a13);
  a62=(a62+a78);
  a80=(a80-a62);
  a62=(a1*a7);
  a80=(a80-a62);
  a80=(a0*a80);
  a62=(a1*a80);
  a79=(a79+a62);
  a62=(a79/a11);
  a6=(a6+a6);
  a31=(a31*a6);
  a62=(a62-a31);
  a31=(1./a11);
  a21=(a21*a6);
  a31=(a31-a21);
  a21=(a31/a11);
  a68=(a68*a6);
  a21=(a21-a68);
  a19=(a19*a21);
  a21=(a9*a15);
  a68=(a1*a73);
  a21=(a21-a68);
  a68=(a12*a82);
  a21=(a21-a68);
  a68=(a14*a80);
  a21=(a21+a68);
  a28=(a28*a21);
  a19=(a19+a28);
  a28=(a14*a82);
  a68=(a9*a73);
  a78=(a1*a15);
  a68=(a68+a78);
  a28=(a28-a68);
  a68=(a12*a80);
  a28=(a28+a68);
  a33=(a33*a28);
  a20=(a20*a6);
  a68=(a20/a11);
  a71=(a71*a6);
  a68=(a68+a71);
  a34=(a34*a68);
  a33=(a33-a34);
  a19=(a19+a33);
  a73=(a12*a73);
  a15=(a14*a15);
  a73=(a73-a15);
  a82=(a1*a82);
  a73=(a73-a82);
  a80=(a9*a80);
  a73=(a73+a80);
  a35=(a35*a73);
  a18=(a18*a6);
  a80=(a18/a11);
  a60=(a60*a6);
  a80=(a80+a60);
  a26=(a26*a80);
  a35=(a35-a26);
  a19=(a19+a35);
  a8=(a8*a79);
  a16=(a16*a6);
  a79=(a16/a11);
  a67=(a67*a6);
  a79=(a79+a67);
  a32=(a32*a79);
  a8=(a8-a32);
  a19=(a19+a8);
  a30=(a30*a19);
  a62=(a62-a30);
  a30=(a9*a20);
  a8=(a12*a18);
  a30=(a30-a8);
  a8=(a14*a16);
  a32=(a1*a31);
  a8=(a8+a32);
  a30=(a30-a8);
  a30=(a0*a30);
  a8=(a14*a30);
  a32=(a1*a20);
  a79=(a12*a16);
  a32=(a32-a79);
  a79=(a9*a31);
  a67=(a14*a18);
  a79=(a79+a67);
  a32=(a32+a79);
  a32=(a0*a32);
  a79=(a12*a32);
  a8=(a8+a79);
  a79=(a1*a18);
  a67=(a9*a16);
  a79=(a79-a67);
  a67=(a14*a20);
  a35=(a12*a31);
  a67=(a67+a35);
  a79=(a79-a67);
  a79=(a0*a79);
  a67=(a9*a79);
  a8=(a8+a67);
  a31=(a14*a31);
  a18=(a9*a18);
  a20=(a12*a20);
  a18=(a18+a20);
  a31=(a31-a18);
  a16=(a1*a16);
  a31=(a31-a16);
  a31=(a0*a31);
  a16=(a1*a31);
  a8=(a8+a16);
  a16=(a8/a3);
  a36=(a36*a59);
  a16=(a16-a36);
  a62=(a62+a16);
  a58=(a58/a3);
  a69=(a69*a59);
  a58=(a58-a69);
  a39=(a39*a58);
  a58=(a9*a32);
  a69=(a1*a30);
  a58=(a58-a69);
  a69=(a12*a79);
  a58=(a58-a69);
  a69=(a14*a31);
  a58=(a58+a69);
  a38=(a38*a58);
  a39=(a39+a38);
  a38=(a14*a79);
  a69=(a9*a30);
  a16=(a1*a32);
  a69=(a69+a16);
  a38=(a38-a69);
  a69=(a12*a31);
  a38=(a38+a69);
  a41=(a41*a38);
  a13=(a13/a3);
  a2=(a2*a59);
  a13=(a13+a2);
  a42=(a42*a13);
  a41=(a41-a42);
  a39=(a39+a41);
  a12=(a12*a30);
  a14=(a14*a32);
  a12=(a12-a14);
  a1=(a1*a79);
  a12=(a12-a1);
  a9=(a9*a31);
  a12=(a12+a9);
  a43=(a43*a12);
  a10=(a10/a3);
  a55=(a55*a59);
  a10=(a10+a55);
  a27=(a27*a10);
  a43=(a43-a27);
  a39=(a39+a43);
  a17=(a17*a8);
  a7=(a7/a3);
  a64=(a64*a59);
  a7=(a7+a64);
  a24=(a24*a7);
  a17=(a17-a24);
  a39=(a39+a17);
  a37=(a37*a39);
  a62=(a62-a37);
  if (res[2]!=0) res[2][7]=a62;
  if (res[2]!=0) res[2][8]=a57;
  a66=(a66/a11);
  a22=(a22/a11);
  a4=(a22*a4);
  a66=(a66-a4);
  a4=(a0*a29);
  a70=(a23*a70);
  a4=(a4+a70);
  a66=(a66-a4);
  a54=(a54/a3);
  a44=(a44/a3);
  a63=(a44*a63);
  a54=(a54-a63);
  a66=(a66+a54);
  a54=(a0*a40);
  a75=(a45*a75);
  a54=(a54+a75);
  a66=(a66-a54);
  if (res[2]!=0) res[2][9]=a66;
  a65=(a65/a11);
  a66=(a22*a5);
  a65=(a65-a66);
  a66=(a23*a77);
  a65=(a65-a66);
  a61=(a61/a3);
  a66=(a44*a56);
  a61=(a61-a66);
  a65=(a65+a61);
  a61=(a45*a74);
  a65=(a65-a61);
  if (res[2]!=0) res[2][10]=a65;
  a73=(a73/a11);
  a22=(a22*a6);
  a73=(a73-a22);
  a23=(a23*a19);
  a73=(a73-a23);
  a12=(a12/a3);
  a44=(a44*a59);
  a12=(a12-a44);
  a73=(a73+a12);
  a45=(a45*a39);
  a73=(a73-a45);
  if (res[2]!=0) res[2][11]=a73;
  if (res[2]!=0) res[2][12]=a76;
  if (res[2]!=0) res[2][13]=a65;
  a81=(a81/a11);
  a25=(a25/a11);
  a5=(a25*a5);
  a81=(a81-a5);
  a5=(a0*a29);
  a77=(a46*a77);
  a5=(a5+a77);
  a81=(a81-a5);
  a51=(a51/a3);
  a48=(a48/a3);
  a56=(a48*a56);
  a51=(a51-a56);
  a81=(a81+a51);
  a51=(a0*a40);
  a74=(a49*a74);
  a51=(a51+a74);
  a81=(a81-a51);
  if (res[2]!=0) res[2][14]=a81;
  a28=(a28/a11);
  a25=(a25*a6);
  a28=(a28-a25);
  a46=(a46*a19);
  a28=(a28-a46);
  a38=(a38/a3);
  a48=(a48*a59);
  a38=(a38-a48);
  a28=(a28+a38);
  a49=(a49*a39);
  a28=(a28-a49);
  if (res[2]!=0) res[2][15]=a28;
  if (res[2]!=0) res[2][16]=a62;
  if (res[2]!=0) res[2][17]=a73;
  if (res[2]!=0) res[2][18]=a28;
  a21=(a21/a11);
  a47=(a47/a11);
  a47=(a47*a6);
  a21=(a21-a47);
  a29=(a0*a29);
  a50=(a50*a19);
  a29=(a29+a50);
  a21=(a21-a29);
  a58=(a58/a3);
  a52=(a52/a3);
  a52=(a52*a59);
  a58=(a58-a52);
  a21=(a21+a58);
  a0=(a0*a40);
  a53=(a53*a39);
  a0=(a0+a53);
  a21=(a21-a0);
  if (res[2]!=0) res[2][19]=a21;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s2;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
