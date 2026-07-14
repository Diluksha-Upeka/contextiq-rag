const puppeteer = require('puppeteer-core');

(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true, // we can run headless in the background
    defaultViewport: { width: 1280, height: 800 }
  });
  const page = await browser.newPage();
  
  console.log("Navigating to http://localhost:5173");
  await page.goto('http://localhost:5173');
  
  console.log("Waiting for Select PDF Document");
  await page.waitForSelector('text=Select PDF Document', { timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));
  
  console.log("Uploading file");
  const inputUploadHandle = await page.$('input[type=file]');
  await inputUploadHandle.uploadFile('test.pdf');
  
  console.log("Waiting for INDEXING IN PROGRESS");
  await page.waitForSelector('text=INDEXING IN PROGRESS', { timeout: 30000 });
  await new Promise(r => setTimeout(r, 3000));
  
  console.log("Taking indexing screenshot");
  await page.screenshot({ path: 'assets/screenshot-indexing.png' });
  
  console.log("Waiting for Document Indexed Successfully");
  await page.waitForSelector('text=Document Indexed Successfully', { timeout: 60000 });
  
  console.log("Waiting for assistant message");
  await page.waitForSelector('text=Document successfully indexed!', { timeout: 30000 });
  
  console.log("Typing question");
  await page.type('textarea', 'What are the key takeaways from this document?');
  
  console.log("Clicking send");
  await page.click('button[type="submit"]');
  
  console.log("Waiting for Context Sources");
  await page.waitForSelector('text=Context Sources', { timeout: 60000 });
  
  console.log("Clicking Context Sources to expand");
  await page.click('text=Context Sources');
  await new Promise(r => setTimeout(r, 1500));
  
  console.log("Taking chat screenshot");
  await page.screenshot({ path: 'assets/screenshot-chat.png' });
  
  console.log("Done");
  await browser.close();
})();
