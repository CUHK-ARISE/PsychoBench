<template>
  <main class="min-h-screen p-10 flex flex-col items-center font-sans text-center bg-white text-black">
    <h1 class="text-5xl font-bold mb-6 mt-4">
      PsychoBench Leaderboard
    </h1>

    <div class="text-xl mb-3 font-semibold flex flex-wrap justify-center gap-x-2">
      <span v-for="(author, i) in authors" :key="i">
        <a
          v-if="author.url"
          :href="author.url"
          class="text-blue-600 hover:underline"
          target="_blank"
          rel="noopener"
        >
          {{ author.name }}
        </a>
        <span v-else>{{ author.name }}</span>
        <sup>{{ author.aff }}</sup><span v-if="i < authors.length - 1">, </span>
      </span>
    </div>

    <div class="text-base text-gray-700 flex justify-center flex-wrap gap-x-6 mb-8">
      <div v-for="(inst, i) in affiliations" :key="i">
        <sup>{{ inst.id }}</sup> {{ inst.name }}
      </div>
    </div>

    <!-- 🔥 Buttons / Badges 區 -->
    <div class="flex flex-wrap justify-center gap-3 mt-6 mb-12">
      <a
        v-for="(link, i) in allLinks"
        :key="i"
        :href="link.url"
        class="transition-transform hover:scale-105"
        target="_blank"
        rel="noopener"
      >
        <img
          :src="link.badge"
          :alt="link.label"
          class="h-7 sm:h-8"
        />
      </a>
    </div>

    <!-- 🖼️ Overview Image -->
    <div class="flex justify-center mt-4 mb-10">
      <img
        src="/overview.png"
        alt="Overview Diagram"
        class="rounded-xl max-w-6xl w-2/3"
      />
    </div>

    <!-- 📊 Leaderboard Iframe -->
    <Leaderboard />


  </main>
</template>

<script setup>
import Leaderboard from '@/pages/Leaderboard.vue'

const authors = [
  { name: "Jen-tse Huang", url: "https://penguinnnnn.github.io", aff: "1,3" },
  { name: "Wenxuan Wang", url: "https://jarviswang94.github.io", aff: "1,3" },
  { name: "Eric John Li", url: "https://ej-l.github.io", aff: "1" },
  { name: "Man Ho Lam", url: "https://donaldlamnl.github.io/DonaldLamNL", aff: "1" },
  { name: "Shujie Ren", url: "", aff: "2" },
  { name: "Youliang Yuan", url: "https://youliangyuan.github.io", aff: "3,4" },
  { name: "Wenxiang Jiao", url: "https://wxjiao.github.io", aff: "3*" },
  { name: "Zhaopeng Tu", url: "https://tuzhaopeng.github.io", aff: "3" },
  { name: "Michael R. Lyu", url: "https://www.cse.cuhk.edu.hk/lyu/home", aff: "1" },
]

const affiliations = [
  { id: 1, name: "The Chinese University of Hong Kong" },
  { id: 2, name: "Institute of Psychology, Tianjin Medical University" },
  { id: 3, name: "Tencent AI Lab" },
  { id: 4, name: "The Chinese University of Hong Kong, Shenzhen" },
]

const allLinks = [
  {
    label: "arXiv Paper",
    url: "https://arxiv.org/pdf/2310.01386",
    badge: "https://img.shields.io/badge/%20arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white",
  },
  {
    label: "GitHub",
    url: "https://github.com/CUHK-ARISE/PsychoBench",
    badge: "https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white",
  },
  {
    label: "ICLR 2024 Oral",
    url: "https://openreview.net/forum?id=H3UayAQWoE",
    badge: "https://img.shields.io/badge/ICLR’24-Oral-success?style=for-the-badge&logo=iclr",
  },
]
</script>

<style scoped>
h1, h2, h3, h4, h5, h6 {
  color: black;
}
</style>