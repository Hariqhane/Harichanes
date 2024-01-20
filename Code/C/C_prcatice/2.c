#include <stdio.h>
#include <stdlib.h>
int main()
{
    int profit ,prize;
    printf("请输入利润：");
    scanf("%d",&profit);
    getchar();
    if(profit>10&&profit<20)
    {
        prize = 10*0.1+(profit-10)*0.075;
    }
    else if(profit>20&&profit<40)
    {
        prize = 10*0.1+10*0.075+(profit-20)*0.05;
    }
    else if(profit>40&&profit<60)
    {
        prize = 10*0.1+10*0.075+20*0.05+(profit - 40)*0.03;
    }
    else if(profit>60&&profit<100)
    {
        prize = 10*0.1+10*0.075+20*0.05+20*0.03+(profit - 60)*0.015;
    }
    else 
    {
        prize = 10*0.1+10*0.075+20*0.05+20*0.03+40*0.015+(profit-100)*0.01;
    }
    printf("奖金为%d万\n",prize);
    getchar();
    return 0;
}


