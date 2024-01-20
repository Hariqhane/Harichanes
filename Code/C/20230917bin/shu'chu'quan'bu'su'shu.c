#include <stdio.h>
int main()
{  
    int x=2;
    int aim;
    scanf("%d",&aim);
    for(;x<=aim;x++){
        int count=2;
        int isPrime =1;
        for(;count<x;count++){
            if(x%count==0){
                isPrime=0;
                break;
                          }      
                             }
            if(isPrime == 1){
                printf("%d\n",x);
                            }
                    }
    system("pause");
    return 0;
}