export interface MenuSettingsInteface {
  menuItem: string
  link?: string
  action: boolean
}

export interface NavListInterface {
  name: string
  status: 'overview' | 'experience' | 'review' | 'settings'
  isActive: boolean
}

export type NavListStatus = 'overview' | 'experience' | 'review' | 'settings'

export interface PagesInteface {
  pageName: string
  link: string
}
