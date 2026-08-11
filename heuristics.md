# Heuristik för att spela Siesta

Det här dokumentet innehåller anteckningar för bra tumregler att följa när man spelar patiensen Siesta.

I dokumentet används orden "stege", "blandad stege" och "färgstege" med följande betydelser:

* Stege: Vilken sekvens som helst med kort i fallande ordning, oavsett färg. Exempelvis 7-6-5.
* Blandad stege: En stege där korten har olika färg.
* Färgstege: En stege där korten har samma färg.

## Regler

* Siesta spelas med en vanlig kortlek med 52 kort.
* Vid start läggs 28 kort uppvända i sju kolumner i en harpform:
  första kolumnen har 7 kort, andra 6, tredje 5 och så vidare ner till 1 kort i den sjunde kolumnen.
* Resterande 24 kort bildar talongen.
* Man får flytta ett toppkort till ett annat kort som är exakt ett högre i valör, oavsett färg.
  Exempel: en ruter 6 får flyttas till en hjärter 7.
* Man får också flytta en toppsekvens som ett enda kort, men bara om den är en sammanhängande färgstege:
  samma färg och fallande med exakt ett steg i taget. Exempel: 6-5-4 i hjärter går att flytta som en enhet, men inte 6-5-3.
* Man får flytta vilken suffix som helst av en giltig färgstege.
  Exempel: om toppen är 6-5-4 i hjärter får man välja att flytta bara 5-4.
* Till ett hål, alltså en tom kolumn, får man flytta vilket toppkort som helst eller vilken giltig färgstege som helst.
* När man inte kan eller vill flytta mer får man dra nya kort från talongen.
  Då läggs ett nytt kort ovanpå varje kolumn, från vänster till höger.
* Om färre än sju kort återstår i talongen delas bara de återstående korten ut, till de första kolumnerna från vänster.
* Alla kort som delats ut ligger kvar öppna; det finns ingen separat foundationhög.
* Man vinner genom att ordna alla fyra färger i kompletta färgstegar från kung ner till ess.
* Ett parti behöver inte ta slut bara för att lagliga drag finns eller saknas.
  I praktiken kan man ha kvar lagliga drag som inte leder framåt, och därför är "inga lagliga drag" inte ett tillförlitligt förlustkriterium.

## Heuristik

* Att göra direkt maximering för att öka stegar eller färgstegar leder i princip aldrig till att patiensen går ut. Istället bör det första målet vara få fram ett hål, eller öka sannolikheten att få fram ett hål.
* När man har ett hål är det värdefullt att leta efter sekvenser av drag som ökar stegar och färgstegar, och gör att man får tillbaka hålet efter förflyttningarna.
  * Det blir lättare att flytta på en blandad stege om det finns andra toppkort som delar av stegen kan mellanlanda på. Med ett hål och en 6:a som toppkort i en annan kolumn, kan en blandad stege 4-5-6-7 flyttas till en 8:a som är toppkort i en tredje kolumn (genom att först flytta 4 till hålet, 5 till 6, 4 till 5, och så vidare -- likt Tornen i Hanoi).
  * Om man kan välja mellan att skapa blandade stegar där många färger eller få färger blandas, är det bättre att blanda få färger. En blandad stege med spader+hjärter och en med ruter+klöver är lättare att ordna upp, än två blandade stegar där alla fyra färger ingår. Men detta bidrar endast på marginalen.
* Att få fram två hål är mycket värdefullt. Det gör det mycket lättare att hitta sekvenser av drag som ökar mängden stegar och färgstegar, utan att hålen förbrukas.
* När man inte kan hitta fler sekvenser av drag som ökar stegar eller färgstegar, utan att hålen förbrukas, kan det vara värt att flytta kort till hålet/hålen innan nya kort läggs upp.
  * Om det går att tydligt öka mängden stegar eller färgstegar genom att förbruka hålen är det i regel bästa valet.
  * Annars är ofta bästa valet att flytta ett kort/toppsekvens till hålet som man har god chans att flytta undan senare, för att få tillbaka hålet, om det samtidigt leder till viss ökning av stegar eller färgstegar.
  * Annars är det oftast bäst att låta hålet vara kvar när nya kort läggs upp.
  * När man väljer kort/toppsekvens att flytta till ett hål, innan nya kort läggs upp, är det bra att titta på vilka valörer som finns kvar i talongen. Om det är stor chans att få upp en 7:a ökar det värdet i att lägga en 6:a i hålet.
* Om tre eller fler kort av samma valör är fastlåsta långt bak i kolumner är det en tydlig riskfaktor. Samma sak gäller om två kort av samma valör är fastlåsta långt bak i en och samma kolumnt. Detta gäller inte för ess och kungar, och bara i viss mån för tvåor och damer. Om många kort av en mittenvalör är fastlåsta kommer det inte att gå att skapa långa stegar, och patiensen riskerar att bli låst.
  * Det är ofta värt att offra hål eller minska mängden stegar och färgstegar för att komma åt sådana fastlåsta kort, eller att komma närmare dem.
* Det är relativt ofarligt om stegar som börjar med kung eller slutar med ess blir fastlåsta under andra kort. Det är alltså av litet (men nollskilt) värde att försöka ta fram sådana stegar.
