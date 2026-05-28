# Boundary Monitor v3 — Portfolio Website

Portfolio site for the Boundary Monitor v3 AI surveillance system.
Deployed on Vercel. The actual engine runs locally / on-device.

## Deployment

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → Login with GitHub
3. Click **Add New Project** → import this repo
4. Click **Deploy** — done

## Customise before deploying

### 1. Add your GitHub link
In `index.html`, replace both instances of:
```
https://github.com/YOUR_USERNAME/boundary-monitor
```

### 2. Add real screenshots
Take screenshots of your running system, save to a `screenshots/` folder,
then in `index.html` replace the placeholder `.screenshot-sim` divs with:
```html
<img src="screenshots/your-file.png" alt="Boundary Monitor HUD" style="width:100%;">
```

### 3. Add demo video
Record your screen (OBS or Win+G), upload to YouTube, then replace the
video placeholder div in `index.html` with:
```html
<iframe width="100%" style="aspect-ratio:16/9;border:1px solid #1a2a1a;"
  src="https://www.youtube.com/embed/YOUR_VIDEO_ID"
  frameborder="0" allowfullscreen></iframe>
```

### 4. Add your resume (optional)
Drop a `resume.pdf` in this folder and add a nav link + button in `index.html`.

## Structure

```
portfolio/
├── index.html      ← entire site (single file)
├── vercel.json     ← Vercel routing config
├── screenshots/    ← add your actual screenshots here
└── README.md
```
