#include <stdio.h>
int main()
{
    int one,two,five;
    int x;
    printf("请输入目标金额：");
    scanf("%d",&x);
    for(one =1;one<=x;one++){

        for(two =1;one+2*two<=x;two++){

            for(five=1;one+2*two+5*five<=x;five++){

                if(one*1+two*2+five*5==x){

                    printf("可以用%d个1元，%d个2元，%d个5元凑成%d元\n",one,two,five,x);
                 
                    goto nihao;
                }
            
            }
        }
    }
    nihao:
    system("pause");
    return 0;
}