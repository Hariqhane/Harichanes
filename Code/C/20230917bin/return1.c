#include <stdio.h>
int max(int a,int b)
{
    int ret;
    if(a>b){
        ret = a;       
    }
     else{
        ret =b ;
    }
}
int main()
{
    int a,b,c;
    scanf("%d %d",&a,&b);
    c=max(a,b);
    printf("%d",c);
    system("pause");
    return 0;
}