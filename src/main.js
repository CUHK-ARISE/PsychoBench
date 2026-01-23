import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './style.css'
import 'uno.css'

const app = createApp(App)
app.use(router)

router.isReady().then(() => {
  /* 只在 GitHub Pages 生效（vite build 後 import.meta.env.PROD === true） */
  if (import.meta.env.PROD) {
    const url = new URL(location.href)
    const redirect = url.searchParams.get('redirect')
    if (redirect) {
      router.replace(redirect)               // 跳到正確路由
      history.replaceState(null, '', redirect) // 清掉 query 參數
    }
  }

  app.mount('#app')
})