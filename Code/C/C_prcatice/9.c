#include <stdio.h>

int main() {
    int n, mul, result = 0,result2=0;
    scanf("%d %d", &n, &mul);
    for(; n > 0; n--) {
        result = result * 10 + mul;
        result2 = result2+result;
        
    }
    printf("%d\n",result2);
    getchar();
    getchar();
    return 0;
}
// #include<stdio.h>
// int main()
// {
//     int s=0,a,n,t;
//     printf("请输入 a 和 n：\n");
//     scanf("%d%d",&a,&n);
//     t=a;
//     while(n>0)
//     {
//         s+=t;
//         a=a*10;
//         t+=a;
//         n--;
//     }
//     printf("a+aa+...=%d\n",s);
//     getchar();
//     getchar();    
//     return 0;
// }
 
