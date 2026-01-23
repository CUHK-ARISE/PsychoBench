import { createRouter, createWebHistory } from 'vue-router'
import LeaderboardPage from '@/pages/Leaderboard.vue'

const routes = [
  { path: '/', component: LeaderboardPage }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router