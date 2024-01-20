#include <stdio.h>
int isPrime(int x , int knowPrimes[],int number0fKnownPrimes);
int main(void)
{
    const int number = 100;
    int prime[number];
    prime[0]=2;//将数组第一个位置填入2
    //此处存在仅限DevC++编译器的写法,即同时定义并初始化数组
    //int prime[number]={2};
    int count = 1;
    int i = 3;
    while (count <number){
        if (isPrime(i,prime,count)){
            prime[count++] = i;
        }
        i++;
    }
    for (i = 0; i < number; i++){
        printf("%d",prime[i]);
        if((i+1)%5) printf("\t");
        else printf("\n");
    }
    getchar();
    return 0;
}
int isPrime(int x , int knowPrimes[],int number0fKnownPrimes)
{
   int ret = 1;
   int i;
   for ( i = 0; i<number0fKnownPrimes;i++){
    if( x % knowPrimes[i] == 0){
        ret = 0;
        break;
    }
   } 
   return ret;
}