/* Stationary Joseph-form innovations, one temporal replicate, double precision.
   No banded inverse, approximation, dynamics integration, or fast-math. */
int _fltused=0;
typedef unsigned long long usize;
void *memset(void *dst,int value,usize n);
void *memcpy(void *dst,const void *src,usize n);
#pragma function(memset,memcpy)
void *memset(void *dst,int value,usize n){volatile unsigned char *p=dst;for(usize i=0;i<n;i++)p[i]=(unsigned char)value;return dst;}
void *memcpy(void *dst,const void *src,usize n){volatile unsigned char *p=dst;const volatile unsigned char *q=src;for(usize i=0;i<n;i++)p[i]=q[i];return dst;}
typedef double (*log_function)(double);
static void zero(double *a,int n){for(int i=0;i<n;i++)a[i]=0.;}
static double *take(double **p,int n){double *q=*p;*p+=n;zero(q,n);return q;}
static void mm(const double*a,const double*b,double*c){for(int i=0;i<5;i++)for(int j=0;j<5;j++){double z=0.;for(int k=0;k<5;k++)z+=a[5*i+k]*b[5*k+j];c[5*i+j]=z;}}
static void mmt(const double*a,const double*b,double*c){for(int i=0;i<5;i++)for(int j=0;j<5;j++){double z=0.;for(int k=0;k<5;k++)z+=a[5*i+k]*b[5*j+k];c[5*i+j]=z;}}
static void triple(const double*a,const double*b,const double*c,double*out,double*tmp){mm(a,b,tmp);mmt(tmp,c,out);}
__declspec(dllexport) int innovations(const double *data,int n,double r,double k,double p1,double p2,double*out,double*workspace,log_function logarithm){
 double *w=workspace;
 double *T=take(&w,25),*dt=take(&w,100),*noise=take(&w,25),*dn=take(&w,100),*cov=take(&w,25),*dc=take(&w,100);
 double *x=take(&w,5),*dx=take(&w,20),*c=take(&w,5),*ddc=take(&w,20),*df=take(&w,4),*de=take(&w,4),*gain=take(&w,5),*dg=take(&w,20);
 double *post=take(&w,5),*dpost=take(&w,20),*J=take(&w,25),*dj=take(&w,100),*pc=take(&w,25),*dpc=take(&w,100),*nc=take(&w,25),*ndc=take(&w,100),*nx=take(&w,5),*ndx=take(&w,20);
 double *tmp=take(&w,25),*a=take(&w,25),*b=take(&w,25),*z=take(&w,25);const double H[5]={1.,-2.,1.,1.,0.};
 double a1=p1*(1-p2),v=(1-p1*p1)*(1-p2*p2);zero(out,10);
 T[0]=r;T[5]=1.;T[11]=1.;T[23]=1.;T[18]=a1;T[19]=p2;
 dt[0]=1.;dt[50+18]=1-p2;dt[75+18]=-p1;dt[75+19]=1.;
 noise[0]=1-r*r;noise[18]=k*v;dn[0]=-2*r;dn[25+18]=k*v;dn[50+18]=-2*k*p1*(1-p2*p2);dn[75+18]=-2*k*p2*(1-p1*p1);
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){int lag=i-j;if(lag<0)lag=-lag;cov[5*i+j]=lag==0?1.:(lag==1?r:r*r);dc[5*i+j]=lag==1?1.:(lag==2?2*r:0.);}
 cov[18]=cov[24]=k;cov[19]=cov[23]=k*p1;for(int i=0;i<25;i++)dc[25+i]=(i>=18?cov[i]:0.);dc[50+19]=dc[50+23]=k;
 for(int t=0;t<n;t++){
  for(int i=0;i<5;i++){c[i]=0.;for(int j=0;j<5;j++)c[i]+=cov[5*i+j]*H[j];}
  double f=0.;for(int i=0;i<5;i++)f+=H[i]*c[i];if(!(f>0.) || f>1e300)return 1;
  for(int h=0;h<4;h++){df[h]=0.;for(int i=0;i<5;i++){ddc[5*h+i]=0.;for(int j=0;j<5;j++)ddc[5*h+i]+=dc[25*h+5*i+j]*H[j];df[h]+=ddc[5*h+i]*H[i];}}
  double e=data[t];for(int i=0;i<5;i++)e-=H[i]*x[i];double ee=e*e;
  for(int h=0;h<4;h++){de[h]=0.;for(int i=0;i<5;i++)de[h]-=H[i]*dx[5*h+i];}
  out[0]+=ee/f;out[1]+=logarithm(f);for(int h=0;h<4;h++){out[2+h]+=2*de[h]*e/f-ee*df[h]/(f*f);out[6+h]+=df[h]/f;}
  for(int i=0;i<5;i++){gain[i]=c[i]/f;post[i]=x[i]+gain[i]*e;for(int h=0;h<4;h++){dg[5*h+i]=ddc[5*h+i]/f-df[h]*c[i]/(f*f);dpost[5*h+i]=dx[5*h+i]+dg[5*h+i]*e+gain[i]*de[h];}}
  for(int i=0;i<5;i++)for(int j=0;j<5;j++){J[5*i+j]=(i==j?1.:0.)-gain[i]*H[j];for(int h=0;h<4;h++)dj[25*h+5*i+j]=-dg[5*h+i]*H[j];}
  triple(J,cov,J,pc,tmp);
  for(int h=0;h<4;h++){triple(dj+25*h,cov,J,a,tmp);triple(J,dc+25*h,J,b,tmp);triple(J,cov,dj+25*h,z,tmp);for(int i=0;i<25;i++)dpc[25*h+i]=a[i]+b[i]+z[i];}
  triple(T,pc,T,nc,tmp);for(int i=0;i<25;i++)nc[i]+=noise[i];
  for(int h=0;h<4;h++){triple(dt+25*h,pc,T,a,tmp);triple(T,dpc+25*h,T,b,tmp);triple(T,pc,dt+25*h,z,tmp);for(int i=0;i<25;i++)ndc[25*h+i]=a[i]+b[i]+z[i]+dn[25*h+i];}
  for(int i=0;i<5;i++){nx[i]=0.;for(int j=0;j<5;j++)nx[i]+=T[5*i+j]*post[j];for(int h=0;h<4;h++){double aa=0.,bb=0.;for(int j=0;j<5;j++){aa+=dt[25*h+5*i+j]*post[j];bb+=T[5*i+j]*dpost[5*h+j];}ndx[5*h+i]=aa+bb;}}
  for(int i=0;i<25;i++)cov[i]=nc[i];for(int i=0;i<100;i++)dc[i]=ndc[i];for(int i=0;i<5;i++)x[i]=nx[i];for(int i=0;i<20;i++)dx[i]=ndx[i];
 }
 return 0;
}
