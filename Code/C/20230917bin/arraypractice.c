#include <stdio.h>
int main()
{
    int x=0;//输入的每个数
    int i=0;//输入数组成数组的位数
    int j;//数字个数组成数组的位数
    int m;//储存输入的数字个数；
    int number1[100];
    int number2[10];
    for (j=0;j<10;j++) {number2[j]=0;}//将数组2初始化
    while(x!=-1){
        scanf("%d",&x);
        number1[i]=x;
        i++;
    }//输入数字储存到数组number1中
    m=i;
    for(i=0;i<=m;i++){
        switch(number1[i]){
            case 0:
                number2[0]++;
                break;
            case 1:
                number2[1]++;
                break;
            case 2:
                number2[2]++;
                break;
            case 3:
                number2[3]++;
                break;
            case 4:
                number2[4]++;
                break;
            case 5:
                number2[5]++;
                break;
            case 6:
                number2[6]++;
                break;
            case 7:
                number2[7]++;
                break;
            case 8:
                number2[8]++;
                break;
            case 9:
                number2[9]++;
                break;
        }
    }//读取数组1存入的每一个数并统计，将统计数字存入数组2
    for(j=0;j<10;j++){
        printf("数字%d出现的次数是:%d\n",j,number2[j]);
    }//读取数组2
    system("pause");
    return 0;
}