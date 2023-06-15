import { MenuSettingsInteface, PagesInteface } from '../interface/HeaderInterface';


export const pages: PagesInteface[] = [
  {
    pageName: 'Home',
    link: '/home'
  },
  {
    pageName: 'Find a Doctor',
    link: '/search-doctor'
  },
  {
    pageName: 'Message',
    link: '/patient/message'
  }
]

export const settings: MenuSettingsInteface[] = [
  {
    menuItem: 'Profile Settings',
    link: '/user/profile-settings',
    action: false
  },
  {
    menuItem: 'Appointments',
    link: '/user/appointments',
    action: false
  },
  {
    menuItem: 'Logout',
    action: true
  }
]

export const settingsAdmin: MenuSettingsInteface[] = [
  {
    menuItem: 'Logout',
    action: true
  }
]