#include <stdio.h>
#include <stdlib.h>
#define Height 100
int main()
{    
    double height = Height; 
    double meter = Height;
    for(int i = 2;i<=10;i++)
    {
        height*=0.5;
        meter+=2*height;
        printf("第%d次落地,反弹高度为%lf,共经过%lf米\n",i,height,meter);
    }
    system("pause");
    return 0;

}