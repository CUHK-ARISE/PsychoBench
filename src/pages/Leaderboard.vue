<template>
  <div class="mt-4 text-gray-700 text-base">
    <div class="mt-10 text-gray-700 text-base">
      <p class="text-md mx-auto leading-relaxed text-justify">
        The leaderboard displays the performance of various models across different psychological questionnaires.
        Select a questionnaire and optionally filter by specific categories to see how models compare.
      </p>
    </div>

    <!-- Questionnaire Buttons -->
    <div class="flex flex-wrap justify-center gap-3 mb-4 mt-6">
      <button
        v-for="questionnaire in questionnaires"
        :key="questionnaire"
        class="px-4 py-2 border rounded-lg transition-colors duration-200 text-sm font-medium"
        :class="{
          'bg-blue-600 text-white': selectedQuestionnaire === questionnaire,
          'bg-white text-gray-700 border-gray-300 hover:bg-gray-100': selectedQuestionnaire !== questionnaire
        }"
        @click="selectQuestionnaire(questionnaire)"
      >
        {{ getQuestionnaireName(questionnaire) }}
      </button>
    </div>

    <!-- Dimension Buttons (only for CABIN) -->
    <div v-if="selectedQuestionnaire === 'CABIN'" class="flex flex-wrap justify-center gap-2 mb-4">
      <button
        v-for="dim in dimensionOptions"
        :key="dim.key"
        class="px-3 py-1.5 border rounded-lg transition-colors duration-200 text-xs font-medium"
        :class="{
          'bg-green-600 text-white': selectedDimension === dim.key,
          'bg-white text-gray-600 border-gray-300 hover:bg-gray-100': selectedDimension !== dim.key
        }"
        @click="selectDimension(dim.key)"
      >
        {{ dim.label }}
      </button>
    </div>

    <!-- Notes Section -->
    <div v-if="selectedQuestionnaire" class="mt-6 mb-6 text-gray-700 text-base max-w-4xl mx-auto">
      <div class="rounded-lg p-4">
        <h2 class="text-3xl font-semibold mb-2">📝 Notes</h2>
        <p class="text-left mb-2">{{ questionnaireDescription }}</p>
        <ul class="list-disc list-inside leading-relaxed text-left space-y-1">
          <li><strong>Scaling:</strong> {{ scoreRangeDisplay }}</li>
          <li><strong>Crowd/Reference:</strong> Baseline scores from human participants (highlighted in yellow).</li>
          <li><strong>Categories:</strong> Subscales or dimensions measured by this questionnaire.</li>
        </ul>
      </div>
    </div>

    <div class="overflow-x-auto shadow-lg rounded-xl bg-white">
      <table class="min-w-full table-auto border-collapse text-lg">
        <thead>
          <tr class="bg-gradient-to-r from-gray-100 to-gray-200 text-base text-gray-700">
            <th
              class="border px-4 py-3 text-center cursor-pointer select-none hover:bg-gray-300 transition-colors duration-150 group"
              @click="toggleSort('model')"
            >
              <div class="h-4 leading-none invisible">▲</div>
              <div class="font-semibold">Model</div>
              <div
                class="text-xs h-4 leading-none transition-opacity duration-150"
                :class="[
                  sortBy === 'model' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40',
                ]"
              >
                {{ sortOrder === 'asc' ? '▲' : '▼' }}
              </div>
            </th>
            <th
              v-for="col in columns"
              :key="col.key"
              class="border px-4 py-2 cursor-pointer select-none hover:bg-gray-300 transition-colors duration-150 text-center group whitespace-nowrap"
              @click="toggleSort(col.key)"
            >
              <div class="h-4 leading-none invisible">▲</div>
              <div class="font-semibold">{{ col.label }}</div>
              <div
                class="text-xs h-4 leading-none transition-opacity duration-150"
                :class="[
                  sortBy === col.key ? 'opacity-100' : 'opacity-0 group-hover:opacity-40',
                ]"
              >
                {{ sortOrder === 'asc' ? '▲' : '▼' }}
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in sortedRows"
            :key="row.model"
            class="hover:bg-blue-50 transition-colors duration-150"
            :class="{ 'bg-yellow-50': row.isCrowd }"
          >
            <td class="border px-4 py-2 font-medium text-gray-800 whitespace-nowrap text-base">
              {{ row.model }}
            </td>
            <td
              v-for="col in columns"
              :key="col.key"
              class="border px-4 py-2 text-center text-gray-700"
            >
              {{ formatCell(row[col.key]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h2 class="text-4xl font-bold text-center mt-12 mb-4">BibTeX</h2>
    <div class="relative w-full max-w-4xl mx-auto">
      <button
        @click="copyBib"
        class="absolute top-2 right-0 flex items-center justify-center rounded bg-gray-200 text-gray-700 hover:bg-gray-300 transition"
        style="width: 85px; height: 30px; font-size: 12px;"
      >
        {{ copied ? 'Copied!' : 'Copy' }}
      </button>
      <div
        v-if="copied"
        class="absolute top-2 right-24 px-3 py-1 bg-green-500 text-white text-sm rounded shadow-lg animate-bounce"
      >
        ✓ Copied!
      </div>
      <pre class="w-full bg-gray-100 p-4 border border-gray-300 text-sm font-mono text-left rounded-xl overflow-x-auto"><code ref="bib">@inproceedings{huang2024humanity,
  title={On the humanity of conversational ai: Evaluating the psychological portrayal of llms},
  author={Huang, Jen-tse and Wang, Wenxuan and Li, Eric John and Lam, Man Ho and Ren, Shujie and \
    Yuan, Youliang and Jiao, Wenxiang and Tu, Zhaopeng and Lyu, Michael},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}</code></pre>
    </div>
    <!-- More Leaderboards Section -->
    <div class="mt-16 mb-8">
      <h2 class="text-4xl font-bold mt-12 mb-4">🤗 More Leaderboards</h2>
      <p class="text-left mb-4">
        Exploring more excellent benchmarks and leaderboards from ARISE Lab:
      </p>
      <ul class="list-disc list-inside leading-relaxed text-left">
        <li>
          <a href="https://cuhk-arise.github.io/GAMABench/" target="_blank" class="text-blue-600 hover:underline">GAMA-Bench Leaderboard</a>
          <span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded bg-orange-100 text-orange-700 border border-orange-300">ICLR'25 Poster</span>
        </li>
        <li>
          <a href="https://cuhk-arise.github.io/EmotionBench/" target="_blank" class="text-blue-600 hover:underline">EmotionBench Leaderboard</a>
          <span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded bg-green-100 text-green-700 border border-green-300">NeurIPS'24 Poster</span>
        </li>
        <li>
          <a href="https://cuhk-arise.github.io/CodeCrash/" target="_blank" class="text-blue-600 hover:underline">CodeCrash Leaderboard</a>
          <span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded bg-green-100 text-green-700 border border-green-300">NeurIPS'25 Poster</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

const leaderboardData = ref({})
const metadata = ref({})
const sortBy = ref('model')
const sortOrder = ref('asc')
const selectedQuestionnaire = ref('')
const selectedDimension = ref('Six Dimensions')
const copied = ref(false)

// Dimension options for CABIN
const dimensionOptions = [
  { key: 'Six Dimensions', label: 'Six Dimensions' },
  { key: 'Eight Dimensions', label: 'Eight Dimensions' }
]

// Get list of questionnaires from data
const questionnaires = computed(() => {
  return Object.keys(leaderboardData.value)
})

// Get full name for a questionnaire key
const getQuestionnaireName = (key) => {
  return metadata.value[key]?.name || key
}

// Get categories for selected questionnaire from metadata
const categories = computed(() => {
  if (!selectedQuestionnaire.value || !metadata.value[selectedQuestionnaire.value]) return []
  const meta = metadata.value[selectedQuestionnaire.value]

  // Special handling for CABIN with dimension filtering
  if (selectedQuestionnaire.value === 'CABIN' && meta[selectedDimension.value]) {
    // Return the dimension group names (e.g., "Six Dimensions D1: Realistic")
    return Object.keys(meta[selectedDimension.value])
  }

  if (meta.categories) {
    return Object.keys(meta.categories)
  }
  return []
})

// Get the mapping from dimension groups to their constituent categories
const dimensionMapping = computed(() => {
  if (selectedQuestionnaire.value !== 'CABIN') return null
  const meta = metadata.value[selectedQuestionnaire.value]
  if (!meta || !meta[selectedDimension.value]) return null
  return meta[selectedDimension.value]
})

// Columns based on categories
const columns = computed(() => {
  return categories.value.map(cat => ({
    key: cat,
    label: cat
  }))
})

// Get score range display for selected questionnaire
const scoreRangeDisplay = computed(() => {
  if (!selectedQuestionnaire.value || !metadata.value[selectedQuestionnaire.value]) return ''
  const meta = metadata.value[selectedQuestionnaire.value]
  const min = meta.min_score
  const max = meta.max_score
  const mode = meta.compute_mode
  const numQuestions = meta.questions ? Object.keys(meta.questions).length : 0

  // Treat EPQ-R as AVG mode
  const treatAsAvg = selectedQuestionnaire.value === 'EPQ-R' || mode === 'AVG'

  if (min !== undefined && max !== undefined) {
    // Show range in parentheses if compute_mode is not AVG
    if (mode && !treatAsAvg && numQuestions > 0) {
      const theoreticalMin = min * numQuestions
      const theoreticalMax = max * numQuestions
      return `Range from ${min} to ${max} for each question (${theoreticalMin}-${theoreticalMax}).`
    }
    return `Range from ${min} to ${max} for each question.`
  }
  return ''
})

// Get questionnaire description
const questionnaireDescription = computed(() => {
  if (!selectedQuestionnaire.value || !metadata.value[selectedQuestionnaire.value]) return ''
  return metadata.value[selectedQuestionnaire.value].description || ''
})

// Select questionnaire
function selectQuestionnaire(questionnaire) {
  selectedQuestionnaire.value = questionnaire
  // Reset dimension to Six Dimensions for CABIN
  if (questionnaire === 'CABIN') {
    selectedDimension.value = 'Six Dimensions'
  }
  // Reset sort
  sortBy.value = 'model'
  sortOrder.value = 'asc'
}

// Select dimension (for CABIN)
function selectDimension(dim) {
  selectedDimension.value = dim
  // Reset sort
  sortBy.value = 'model'
  sortOrder.value = 'asc'
}

// Determine if a model name represents crowd/reference data
function isCrowdModel(modelName) {
  const crowdPatterns = ['Crowd', 'Male', 'Female', 'Men', 'Women', 'Students', 'Workers', 'USA', 'Whole Sample']
  return crowdPatterns.some(pattern => modelName.includes(pattern))
}

// Compute row data for each model
const sortedRows = computed(() => {
  if (!selectedQuestionnaire.value) return []

  const questionnaireData = leaderboardData.value[selectedQuestionnaire.value] || {}

  const rows = Object.entries(questionnaireData).map(([model, scores]) => {
    const row = { model, isCrowd: isCrowdModel(model) }

    // For CABIN with dimension groups, compute average of constituent categories
    if (selectedQuestionnaire.value === 'CABIN' && dimensionMapping.value) {
      for (const dimGroup of categories.value) {
        const constituentCategories = dimensionMapping.value[dimGroup] || []
        let sum = 0
        let count = 0
        for (const cat of constituentCategories) {
          if (scores[cat] != null) {
            sum += scores[cat]
            count++
          }
        }
        row[dimGroup] = count > 0 ? sum / count : null
      }
    } else {
      // Regular categories
      for (const cat of categories.value) {
        row[cat] = scores[cat] ?? null
      }
    }
    return row
  })

  // Sort by selected column
  return rows.sort((a, b) => {
    if (sortBy.value === 'model') {
      const cmp = a.model.localeCompare(b.model)
      return sortOrder.value === 'asc' ? cmp : -cmp
    }
    const aVal = a[sortBy.value]
    const bVal = b[sortBy.value]
    if (aVal == null) return 1
    if (bVal == null) return -1
    return sortOrder.value === 'asc' ? aVal - bVal : bVal - aVal
  })
})

function toggleSort(column) {
  if (sortBy.value === column) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortBy.value = column
    sortOrder.value = 'desc'
  }
}

function formatCell(val) {
  if (val == null) return '-'
  return typeof val === 'number' ? val.toFixed(2) : val
}

const bib = ref(null)

function copyBib() {
  if (bib.value) {
    navigator.clipboard.writeText(bib.value.textContent)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  }
}

onMounted(async () => {
  try {
    const [dataRes, metaRes] = await Promise.all([
      fetch(`${import.meta.env.BASE_URL}data.json`),
      fetch(`${import.meta.env.BASE_URL}metadata.json`)
    ])
    leaderboardData.value = await dataRes.json()
    metadata.value = await metaRes.json()

    // Initialize with first questionnaire selected
    if (questionnaires.value.length > 0) {
      selectQuestionnaire(questionnaires.value[0])
    }
  } catch (e) {
    console.error('Failed to load data:', e)
  }
})
</script>

<style scoped>
table {
  font-family: Arial, sans-serif;
}
</style>
